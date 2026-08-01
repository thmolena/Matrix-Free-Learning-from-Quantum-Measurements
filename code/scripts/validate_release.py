#!/usr/bin/env python3
"""Validate the public release structure and delegate to the scientific validator."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CODE = REPOSITORY / "code"
REQUIRED_ROOT = {".gitignore", "LICENSE", "README.md", "index.html", "main.tex", "main.pdf", "code"}
REQUIRED_CODE = {"README.md", "pyproject.toml", "src", "tests", "scripts", "configs", "data", "results", "manuscript_assets"}
DELEGATE = ["cptppinn-validate"]


def main() -> None:
    root_names = {p.name for p in REPOSITORY.iterdir() if p.name != ".git"}
    missing_root = sorted(REQUIRED_ROOT - root_names)
    extra_root = sorted(root_names - REQUIRED_ROOT)
    code_names = {p.name for p in CODE.iterdir()}
    missing_code = sorted(REQUIRED_CODE - code_names)
    if missing_root or extra_root or missing_code:
        raise SystemExit(
            json.dumps(
                {"missing_root": missing_root, "extra_root": extra_root, "missing_code": missing_code},
                indent=2,
            )
        )
    manuscript = (REPOSITORY / "main.tex").read_text(encoding="utf-8")
    if not manuscript.startswith(r"\documentclass[10pt,letterpaper,twoside]{article}"):
        raise SystemExit("one-column preprint class contract changed")
    if DELEGATE:
        command = list(DELEGATE)
        environment = os.environ.copy()
        if command[0] == "python":
            command[0] = sys.executable
        elif shutil.which(command[0]) is None:
            command = [sys.executable, "-m", "ctpcpinn.validate"]
            source = str(CODE / "src")
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (source, environment.get("PYTHONPATH", "")) if part
            )
        subprocess.run(command, cwd=REPOSITORY, check=True, env=environment)
    print("release contract: PASS")


if __name__ == "__main__":
    main()
