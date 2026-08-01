"""Evidence figures for safety-gated Ramsey reconstruction."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def main() -> None:
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(); parser.add_argument("--study",type=Path,default=Path("code/results/adaptive_ramsey.json")); parser.add_argument("--output",type=Path,default=Path("code/manuscript_assets/figures/adaptive")); args=parser.parse_args()
    d=json.loads(args.study.read_text()); args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":8.2,"axes.labelsize":8.2,"axes.titlesize":8.2,"legend.fontsize":7,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#dddddd","grid.linewidth":.5,"legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":"tight"})
    blue,orange,green,ink="#0072B2","#D55E00","#009E73","#222222"; rows=d["records"]
    fig,axes=plt.subplots(2,2,figsize=(7,5.0))
    states=list(d["by_state"]); x=np.arange(len(states));
    axes[0,0].plot(x,[d["by_state"][s]["linear_mean_rmse"] for s in states],"-o",color=ink,label="linear")
    axes[0,0].plot(x,[d["by_state"][s]["adaptive_mean_rmse"] for s in states],"-o",color=blue,label="safety-gated")
    axes[0,0].set(xticks=x,xticklabels=states,ylabel="mean held-out RMSE");axes[0,0].legend();axes[0,0].text(-.14,1.04,"(a)",transform=axes[0,0].transAxes,fontweight="bold")
    ordered=sorted(rows,key=lambda r:r["linear_test_rmse"]); j=np.arange(len(ordered))
    axes[0,1].plot(j,[r["linear_test_rmse"] for r in ordered],color=ink,label="linear")
    axes[0,1].plot(j,[r["adaptive_test_rmse"] for r in ordered],color=blue,label="safety-gated")
    axes[0,1].set(xlabel="trace (sorted)",ylabel="held-out RMSE");axes[0,1].legend();axes[0,1].text(-.14,1.04,"(b)",transform=axes[0,1].transAxes,fontweight="bold")
    ratio=np.array([r["best_spectral_development_rmse"]/r["linear_development_rmse"] for r in rows]); delta=np.array([r["linear_test_rmse"]-r["adaptive_test_rmse"] for r in rows]); sel=np.array([r["selected"]=="spectral" for r in rows])
    axes[1,0].scatter(ratio[~sel],delta[~sel],s=10,color="#888888",alpha=.7,label="linear retained")
    axes[1,0].scatter(ratio[sel],delta[sel],s=12,color=green,alpha=.75,label="spectral accepted")
    axes[1,0].axvline(.95,color=orange,ls="--");axes[1,0].axhline(0,color=ink,lw=.6);axes[1,0].set(xlabel="development spectral / linear RMSE",ylabel="outer-test RMSE gain");axes[1,0].legend();axes[1,0].text(-.14,1.04,"(c)",transform=axes[1,0].transAxes,fontweight="bold")
    b=d["runtime"]["benchmark"]; count=[r["traces"] for r in b]
    axes[1,1].plot(count,[r["matrix_free_seconds"] for r in b],"-o",color=green,label="block solve")
    axes[1,1].plot(count,[r["dense_solve_seconds"] for r in b],"-s",color=orange,label="explicit solve")
    axes[1,1].set(xlabel="number of traces",ylabel="median solve time (s)",yscale="log");axes[1,1].legend();axes[1,1].text(-.14,1.04,"(d)",transform=axes[1,1].transAxes,fontweight="bold")
    fig.subplots_adjust(hspace=.38,wspace=.30);fig.savefig(args.output/"adaptive_ramsey.pdf");fig.savefig(args.output/"adaptive_ramsey.png",dpi=220);plt.close(fig)


if __name__ == "__main__": main()
