# scripts/plot_miso_sim_compare.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_hist(csv_path: str | Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    # accept typical gen column names
    gen_col = None
    for c in ["Gen", "generation", "gen_mwh", "mwh"]:
        if c in df.columns:
            gen_col = c
            break
    if gen_col is None:
        raise ValueError("No Gen column found in historical CSV.")
    df = df[["Datetime", gen_col]].dropna().sort_values("Datetime")
    df = df.set_index("Datetime")
    return pd.to_numeric(df[gen_col], errors="coerce").dropna().rename("Hist_Gen")

def load_sim(csv_path: str | Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    # accept either 'Gen' or a named column
    gen_col = "Gen" if "Gen" in df.columns else None
    if gen_col is None:
        # try to find any non-Datetime numeric column
        for c in df.columns:
            if c != "Datetime" and np.issubdtype(df[c].dtype, np.number):
                gen_col = c; break
    if gen_col is None:
        raise ValueError("No numeric generation column found in simulation CSV.")
    df = df[["Datetime", gen_col]].dropna().sort_values("Datetime")
    df = df.set_index("Datetime")
    return pd.to_numeric(df[gen_col], errors="coerce").dropna().rename("Sim_Gen")

def main():
    ap = argparse.ArgumentParser(description="Compare historical MISO Gen vs simulated hourly Gen (separate + overlap views).")
    ap.add_argument("hist_csv", type=str, help="Path to MISO_cleaned.csv (Datetime, Gen).")
    ap.add_argument("sim_csv", type=str, help="Path to simulated CSV (Datetime, Gen).")
    ap.add_argument("--overlap_len_hours", type=int, default=24*30, help="Length of series to show in overlap (by index), default 30 days.")
    ap.add_argument("--overlap_smooth_h", type=int, default=6, help="Rolling mean hours for overlap smoothing (0 = no smoothing).")
    ap.add_argument("--out_png", type=str, default="plots/miso_hist_vs_sim.png")
    ap.add_argument("--show", action="store_true", help="Show the plot window.")
    args = ap.parse_args()

    hist = load_hist(args.hist_csv)
    sim  = load_sim(args.sim_csv)

    # Ensure hourly frequency labels (not strictly required for plotting)
    hist = hist.sort_index()
    sim  = sim.sort_index()

    # Build figure
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True, sharex=False)
    fig.suptitle("MISO Generation: Historical vs Simulated (Hourly)", fontsize=14)

    # 1) Historical (full range)
    axes[0].plot(hist.index, hist.values, linewidth=0.8)
    axes[0].set_title("Historical (from dataset)")
    axes[0].set_ylabel("Gen")
    axes[0].grid(alpha=0.3)

    # 2) Simulated (full range)
    axes[1].plot(sim.index, sim.values, linewidth=0.8)
    axes[1].set_title("Simulated (2026–2030)")
    axes[1].set_ylabel("Gen")
    axes[1].grid(alpha=0.3)

    # 3) Overlap view (index-aligned)
    # pick a recent slice of historical for a comparable window length
    N = min(args.overlap_len_hours, len(hist), len(sim))
    if N <= 0:
        raise ValueError("Not enough data to create overlap view. Try lowering --overlap_len_hours.")
    hist_seg = hist.iloc[-N:].reset_index(drop=True)
    sim_seg  = sim.iloc[:N].reset_index(drop=True)

    if args.overlap_smooth_h and args.overlap_smooth_h > 1:
        hist_seg = hist_seg.rolling(args.overlap_smooth_h, min_periods=1).mean()
        sim_seg  = sim_seg.rolling(args.overlap_smooth_h,  min_periods=1).mean()

    x = np.arange(N)  # hour index
    axes[2].plot(x, hist_seg.values, label="Historical (last N hours)", linewidth=0.9)
    axes[2].plot(x, sim_seg.values,  label="Simulated (first N hours)", linewidth=0.9)
    axes[2].set_title(f"Overlap (Index-Aligned) — N={N} hours"
                      + (f", {args.overlap_smooth_h}h rolling mean" if args.overlap_smooth_h and args.overlap_smooth_h > 1 else ""))
    axes[2].set_xlabel("Hour index (0…N)")
    axes[2].set_ylabel("Gen")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.savefig(args.out_png, dpi=200)
    print(f"Saved: {Path(args.out_png).resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
