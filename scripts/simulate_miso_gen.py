# scripts/simulate_miso_gen.py
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# -----------------------
# Helpers: loading & checks
# -----------------------
def load_gen(csv_path: str | Path, dup_strategy: str = "mean") -> pd.Series:
    """
    Load an hourly generation time series from CSV and return a clean, monotonic Series indexed by timestamp.
    Handles duplicate timestamps (e.g., DST fallback) via `dup_strategy`: one of {"mean","sum","first"}.
    """
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    if "Gen" not in df.columns:
        for col in ("generation", "gen_mwh", "mwh"):
            if col in df.columns:
                df["Gen"] = pd.to_numeric(df[col], errors="coerce")
                break
        else:
            raise ValueError("No 'Gen' column found (looked for: Gen, generation, gen_mwh, mwh).")
    else:
        df["Gen"] = pd.to_numeric(df["Gen"], errors="coerce")

    df = df.dropna(subset=["Datetime", "Gen"]).sort_values("Datetime").set_index("Datetime")

    if df.index.has_duplicates:
        if dup_strategy == "mean":
            df = df.groupby(level=0, sort=True).agg({"Gen": "mean"})
        elif dup_strategy == "sum":
            df = df.groupby(level=0, sort=True).agg({"Gen": "sum"})
        elif dup_strategy == "first":
            df = df[~df.index.duplicated(keep="first")].sort_index()
        else:
            raise ValueError("dup_strategy must be one of: 'mean', 'sum', 'first'")

    df = df.sort_index()
    idx_full = pd.date_range(df.index.min(), df.index.max(), freq="H")
    df = df.reindex(idx_full)
    df["Gen"] = df["Gen"].interpolate(limit=3).fillna(method="bfill").fillna(method="ffill")
    return df["Gen"].clip(lower=0.0)


def parse_year_multipliers(s: str) -> dict[int, float]:
    """
    Parse like: "2026:1.05,2027:1.10,2028:1.10,2029:1.12,2030:1.15"
    """
    out: dict[int, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        y, m = part.split(":")
        out[int(y.strip())] = float(m.strip())
    return out

# -----------------------
# Core: month-aware bootstrap
# -----------------------
def month_window(m: int, width: int = 1) -> set[int]:
    """
    Return a set of months around m, +/- width (wrap around year).
    width=1 -> {m-1, m, m+1}
    """
    out = set()
    for d in range(-width, width + 1):
        mm = ((m - 1 + d) % 12) + 1
        out.add(mm)
    return out

def sample_month_blocks(gen: pd.Series, year: int, block_len=6, month_band=1,
                        rng: np.random.Generator | None = None) -> pd.Series:
    """
    Build a synthetic hourly series for 'year' by stitching together
    contiguous blocks sampled from historical hours that fall in a small
    month window around the target month.

    - gen: historical hourly Gen (pd.Series, hourly DatetimeIndex)
    - year: target calendar year
    - block_len: hours per block (6–12 is typical)
    - month_band: how wide the month window is (1 = target month +/-1)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-split history by month for quick sampling
    hist = gen.dropna()
    hist_by_month = {m: hist[hist.index.month == m] for m in range(1, 13)}

    # Target hourly index for the whole year (UTC-naive)
    start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    end   = pd.Timestamp(year=year, month=12, day=31, hour=23)
    target_idx = pd.date_range(start, end, freq="h")

    out_vals = []
    pos = 0
    while pos < len(target_idx):
        target_ts = target_idx[pos]
        target_month = target_ts.month
        months_pool = month_window(target_month, width=month_band)

        # Build candidate start indices from pooled months
        candidates = []
        for m in months_pool:
            ser = hist_by_month[m]
            # only start positions that have at least block_len contiguous hours ahead in SAME month subset
            # We'll just ensure index positions exist contiguously in the global series
            ser_idx = ser.index.view("int64")
            # We map to the global history index to pull contiguous blocks
            # Find corresponding positions in the full 'hist' index
            # For speed, precompute a dict
            # (Do on first call only)
            # We'll approximate by sampling starts from the month's hours and trusting short gaps are rare
            candidates.extend(ser.index)

        if not candidates:
            # fallback: sample anywhere in history
            candidates = list(hist.index)

        # pick a random start, then take a contiguous block from the full series
        start_ts = rng.choice(candidates)
        # Align to hour
        start_ts = pd.Timestamp(start_ts.floor("h"))
        # Build block from the global series (not month-filtered) to ensure contiguity
        # Find nearest location in the full series
        try:
            loc = hist.index.get_loc(start_ts)
            if isinstance(loc, slice):
                # slice returned (duplicate timestamps) – simplify
                loc = hist.index.slice_indexer(start_ts, start_ts).start
        except KeyError:
            # If start_ts not in index (unlikely), just move on
            continue

        end_loc = min(loc + block_len, len(hist))
        block = hist.iloc[loc:end_loc].values
        if len(block) < block_len:
            # try again if too short
            continue

        out_needed = len(target_idx) - pos
        take = min(block_len, out_needed)
        out_vals.extend(block[:take])
        pos += take

    synth = pd.Series(np.array(out_vals, dtype=float), index=target_idx, name="Gen")
    return synth.clip(lower=0.0)

# -----------------------
# Capacity scaling
# -----------------------
def capacity_scale_for_years(years: list[int],
                             base_mult: float = 1.0,
                             cagr: float | None = None,
                             per_year_mult: dict[int, float] | None = None) -> dict[int, float]:
    """
    Compute scaling multipliers for each year.
    Priority:
    1) per_year_mult if provided (absolute multipliers)
    2) CAGR relative to first year (mult = (1+CAGR)^(year - start_year))
    3) base_mult (constant)
    """
    out = {}
    per_year_mult = per_year_mult or {}
    y0 = min(years)
    for y in years:
        if y in per_year_mult:
            out[y] = per_year_mult[y]
        elif cagr is not None:
            out[y] = base_mult * ((1.0 + cagr) ** (y - y0))
        else:
            out[y] = base_mult
    return out

# -----------------------
# Driver
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Simulate hourly MISO generation 2026–2030 via month-aware block bootstrap + capacity scaling.")
    ap.add_argument("input_csv", type=str, help="Path to MISO_cleaned.csv (must include Datetime, Gen).")
    ap.add_argument("--start_year", type=int, default=2026)
    ap.add_argument("--end_year", type=int, default=2030)
    ap.add_argument("--block_len", type=int, default=6, help="Hours per sampled block (typical 6–12).")
    ap.add_argument("--month_band", type=int, default=1, help="Month window width: 1 uses {m-1, m, m+1}.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sims", type=int, default=1, help="Number of Monte Carlo simulations to produce.")
    ap.add_argument("--out_dir", type=str, default="projections")
    # Capacity scaling
    ap.add_argument("--cagr", type=float, default=None, help="Annual growth rate as decimal (e.g., 0.06 for 6%%).")
    ap.add_argument("--base_mult", type=float, default=1.0, help="Base multiplier (used if CAGR or per-year not given).")
    ap.add_argument("--per_year_mult", type=str, default="", help='Year multipliers like "2026:1.05,2027:1.10,2028:1.10,2029:1.12,2030:1.15" (overrides CAGR).')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gen_hist = load_gen(args.input_csv)

    years = list(range(args.start_year, args.end_year + 1))
    year_mults = capacity_scale_for_years(
        years,
        base_mult=args.base_mult,
        cagr=args.cagr,
        per_year_mult=parse_year_multipliers(args.per_year_mult)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in range(args.n_sims):
        # fresh RNG per sim for reproducibility with different seeds
        sim_rng = np.random.default_rng(args.seed + s)
        parts = []
        for y in years:
            sim_y = sample_month_blocks(gen_hist, y, block_len=args.block_len,
                                        month_band=args.month_band, rng=sim_rng)
            scale = year_mults[y]
            sim_y = (sim_y * scale).rename("Gen")
            parts.append(sim_y)

        sim = pd.concat(parts).to_frame()
        sim["Sim"] = s + 1
        # Save one file per simulation
        out_csv = out_dir / f"MISO_sim_hourly_{years[0]}_{years[-1]}_sim{s+1}.csv"
        sim.to_csv(out_csv, index_label="Datetime")
        print(f"Saved: {out_csv.resolve()} (hours={len(sim)})")

if __name__ == "__main__":
    main()
