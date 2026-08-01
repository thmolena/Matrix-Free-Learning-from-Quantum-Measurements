"""Fetch and verify the official Zenodo Ramsey archive, then derive the CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import urllib.request
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .real_data import DERIVED_CSV_NAME, RAMSEY_ARCHIVE_SHA256


RAMSEY_URL = (
    "https://zenodo.org/api/records/15797402/files/Ramsey.zip/content"
)
RAMSEY_MD5 = "c78b673a91a93399b1b4f13699f31ce8"
SHEET = "xl/worksheets/sheet1.xml"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DELAY_RE = re.compile(r"delay_([0-9.]+)_?ms")
STEP_RE = re.compile(r"step_([0-9]+)")
REPETITIONS_RE = re.compile(r"repetitions_([0-9]+)")


def _numeric_cells(workbook_bytes: bytes) -> list[float]:
    with ZipFile(io.BytesIO(workbook_bytes)) as workbook:
        root = ET.fromstring(workbook.read(SHEET))
    return [
        float(value.text)
        for value in root.findall(f".//{NS}v")
        if value.text is not None
    ]


def derive_csv(archive_path: Path, output_path: Path) -> dict:
    archive_bytes = archive_path.read_bytes()
    sha256 = hashlib.sha256(archive_bytes).hexdigest()
    md5 = hashlib.md5(archive_bytes).hexdigest()  # nosec: archive identity only
    if sha256 != RAMSEY_ARCHIVE_SHA256:
        raise ValueError(f"Ramsey.zip SHA-256 mismatch: {sha256}")
    if md5 != RAMSEY_MD5:
        raise ValueError(f"Ramsey.zip Zenodo MD5 mismatch: {md5}")

    fieldnames = [
        "trace_id",
        "state",
        "delay_ms",
        "phase_step_deg",
        "phase_index",
        "phase_deg",
        "excitation_probability",
        "repetitions",
        "source_file",
        "source_archive_sha256",
    ]
    trace_id = 0
    observations = 0
    with ZipFile(io.BytesIO(archive_bytes)) as archive, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(
            item for item in archive.namelist() if item.lower().endswith(".xlsx")
        ):
            parts = Path(name).parts
            state = parts[-2]
            filename = parts[-1]
            delay_match = DELAY_RE.search(filename)
            step_match = STEP_RE.search(filename)
            if delay_match is None or step_match is None:
                raise ValueError(f"cannot parse Ramsey metadata from {name}")
            delay_ms = float(delay_match.group(1))
            step = int(step_match.group(1))
            repetition_match = REPETITIONS_RE.search(filename)
            repetitions = int(repetition_match.group(1)) if repetition_match else 100
            values = _numeric_cells(archive.read(name))
            for phase_index, probability in enumerate(values):
                writer.writerow({
                    "trace_id": trace_id,
                    "state": state,
                    "delay_ms": f"{delay_ms:.10g}",
                    "phase_step_deg": step,
                    "phase_index": phase_index,
                    "phase_deg": phase_index * step,
                    "excitation_probability": f"{probability:.17g}",
                    "repetitions": repetitions,
                    "source_file": name,
                    "source_archive_sha256": sha256,
                })
                observations += 1
            trace_id += 1
    return {
        "archive_sha256": sha256,
        "archive_md5": md5,
        "derived_csv_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "traces": trace_id,
        "observations": observations,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the official experimental Ramsey archive and derive CSV."
    )
    parser.add_argument("--archive", default=None, help="use an existing Ramsey.zip")
    parser.add_argument("--output", default=DERIVED_CSV_NAME)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.archive is None:
        archive = output.with_name("Ramsey.zip")
        with urllib.request.urlopen(RAMSEY_URL, timeout=120) as response:
            archive.write_bytes(response.read())
    else:
        archive = Path(args.archive)
    audit = derive_csv(archive, output)
    for key, value in audit.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
