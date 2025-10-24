#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Robust readers (CSV/Parquet) with datetime normalization
# -----------------------
DT_CANDIDATES = ("datetime", "time", "timestamp", "date", "dt")

def _normalize_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Find a datetime-like column (case-insensitive), parse to datetime, and rename to 'Datetime'."""
    lower = {c.lower(): c for c in df.columns}
    for cand in DT_CANDIDATES:
        if cand in lower:
            col = lower[cand]
            if col != "Datetime":
                df = df.rename(columns={col: "Datetime"})
            # Parse; coerce bad rows to NaT (will be dropped)
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
            return df
    # If an exact 'Datetime' exists, parse it
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        return df
    raise ValueError("No Datetime/Time/Timestamp column found.")

def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet, then normalize datetime column name & dtype."""
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return _normalize_datetime_column(df)
    if ext in (".csv", ".txt"):
        # Read quickly, then normalize and (re)parse date col only once
        df = pd.read_csv(path)
        return _normalize_datetime_column(df)
    raise ValueError(f"Unsupported file type: {path}")

# -----------------------
# Column picking helpers
# -----------------------
def _pick_gen_col(df: pd.DataFrame) -> str:
    for c in ["Gen", "generation", "gen_mwh", "mwh"]:
        if c in df.columns:
            return c
    raise ValueError("Historical file must have 'Gen' (or generation/gen_mwh/mwh).")

def _pick_sim_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns and np.issubdtype(df[preferred].dtype, np.number):
        return preferred
    sim_like = [c for c in df.columns if c != "Datetime" and c.lower().startswith("sim_") and np.issubdtype(df[c].dtype, np.number)]
    if sim_like:
        return sim_like[0]
    # fallback: first numeric non-Datetime column
    for c in df.columns:
        if c != "Datetime" and np.issubdtype(df[c].dtype, np.number):
            return c
    raise ValueError("No numeric simulation column found.")

# -----------------------
# Loaders
# -----------------------
def load_hist(csv_path: str | Path) -> pd.Series:
    df = _read_any(Path(csv_path))
    gen_col = _pick_gen_col(df)
    df = df.dropna(subset=["Datetime", gen_col]).sort_values("Datetime")
    s = df.set_index("Datetime")[gen_col].astype(float).rename("Hist_Gen")
    # Ensure DatetimeIndex
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].sort_index()
    return s

def load_sim(sim_path: str | Path, sim_col: str | None = None) -> pd.Series:
    p = Path(sim_path)
    if p.is_dir():
        files = sorted(list(p.glob("hourly_sims_*.parquet")) + list(p.glob("hourly_sims_*.csv")))
        if not files:
            raise ValueError(
                f"No hourly sims found in {p}. Expected files like hourly_sims_*.parquet or hourly_sims_*.csv."
            )
        chosen = None
        parts = []
        for f in files:
            df = _read_any(f)
            if "Datetime" not in df.columns:
                raise ValueError(f"{f} has no Datetime column.")
            if chosen is None:
                chosen = _pick_sim_col(df, sim_col)
            if chosen not in df.columns:
                # If requested sim_col missing in this chunk, try to pick an available numeric sim column
                alt = _pick_sim_col(df, None)
                if sim_col:
                    raise ValueError(f"Requested sim_col '{sim_col}' not in chunk {f}. Has: {list(df.columns)}")
                chosen = alt
            parts.append(df[["Datetime", chosen]])
        df_all = pd.concat(parts, ignore_index=True)
        df_all = df_all.dropna(subset=["Datetime"]).sort_values("Datetime").drop_duplicates("Datetime", keep="last")
        s = df_all.set_index("Datetime")[chosen].astype(float).rename("Sim_Gen")
    else:
        # Friendly guard: user accidentally passed the monthly summary
        if p.name.startswith("monthly_quantiles"):
            raise ValueError(
                "You passed a monthly summary CSV. Use an hourly file (hourly_sims_*.parquet/.csv) "
                "or the folder containing those files."
            )
        df = _read_any(p)
        if "Datetime" not in df.columns:
            raise ValueError("Simulation file must include a Datetime/Time/Timestamp column.")
        col = _pick_sim_col(df, sim_col)
        df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
        s = df.set_index("Datetime")[col].astype(float).rename("Sim_Gen")
    # Ensure DatetimeIndex
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].sort_index()
    return s

# -----------------------
# Plot
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Compare CAISO historical vs simulated hourly generation.")
    ap.add_argument("hist_csv", type=str, help="Path to CAISO_cleaned.csv (must include Datetime + Gen).")
    ap.add_argument("sim_input", type=str, help="Path to hourly sims file (CSV/Parquet) OR directory with hourly_sims_* files.")
    ap.add_argument("--sim_col", type=str, default=None, help="Simulation column to plot (e.g., sim_0).")
    ap.add_argument("--overlap_len_hours", type=int, default=24*30, help="Overlap length in hours (default 30 days).")
    ap.add_argument("--overlap_smooth_h", type=int, default=6, help="Rolling mean hours for the overlap panel (0 = none).")
    ap.add_argument("--title", type=str, default="CAISO Solar: Historical vs Simulated (Hourly)")
    ap.add_argument("--out_png", type=str, default="plots/caiso_hist_vs_sim.png")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")
    args = ap.parse_args()

    hist = load_hist(args.hist_csv).sort_index()
    sim  = load_sim(args.sim_input, sim_col=args.sim_col).sort_index()

    # Double-safety: ensure DatetimeIndex for both
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index, errors="coerce")
        hist = hist[~hist.index.isna()].sort_index()
    if not isinstance(sim.index, pd.DatetimeIndex):
        sim.index = pd.to_datetime(sim.index, errors="coerce")
        sim = sim[~sim.index.isna()].sort_index()

    # Prepare figure
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True, sharex=False)
    fig.suptitle(args.title, fontsize=14)

    # 1) Historical panel
    axes[0].plot(hist.index, hist.values, linewidth=0.8)
    axes[0].set_title("Historical (from dataset)")
    axes[0].set_ylabel("MW")
    axes[0].grid(alpha=0.3)

    # 2) Simulated panel with inferred horizon
    axes[1].plot(sim.index, sim.values, linewidth=0.8)
    try:
        years = pd.Index(sim.index).year
        label = f"Simulated ({years.min()}–{years.max()})" if len(years) else "Simulated"
    except Exception:
        label = "Simulated"
    axes[1].set_title(label)
    axes[1].set_ylabel("MW")
    axes[1].grid(alpha=0.3)

    # 3) Overlap panel (index-aligned)
    N = min(args.overlap_len_hours, len(hist), len(sim))
    if N <= 0:
        raise ValueError("Not enough data to build an overlap panel. Try lowering --overlap_len_hours.")
    hist_seg = hist.iloc[-N:].reset_index(drop=True)
    sim_seg  = sim.iloc[:N].reset_index(drop=True)

    if args.overlap_smooth_h and args.overlap_smooth_h > 1:
        hist_seg = hist_seg.rolling(args.overlap_smooth_h, min_periods=1).mean()
        sim_seg  = sim_seg.rolling(args.overlap_smooth_h,  min_periods=1).mean()

    x = np.arange(N)
    axes[2].plot(x, hist_seg.values, label="Historical (last N hours)", linewidth=0.9)
    axes[2].plot(x, sim_seg.values,  label="Simulated (first N hours)", linewidth=0.9)
    subtitle = f"Overlap (Index-Aligned) — N={N} hours"
    if args.overlap_smooth_h and args.overlap_smooth_h > 1:
        subtitle += f", {args.overlap_smooth_h}h rolling mean"
    axes[2].set_title(subtitle)
    axes[2].set_xlabel("Hour index (0…N)")
    axes[2].set_ylabel("MW")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()
