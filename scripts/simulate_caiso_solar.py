#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


# -----------------------
# Load & clean historical generation
# -----------------------
def load_gen(csv_path: str | Path) -> pd.Series:
    """
    Load a CSV with columns ['Datetime', 'Gen'] (or common aliases).
    - Parses timestamps
    - Ensures a strictly hourly, gap-free index (forward/back fill tiny gaps)
    - Returns a clean hourly pd.Series named 'Gen' (MW)
    """
    df = pd.read_csv(csv_path)

    # Pick a generation column
    gen_col = None
    for c in ["Gen", "generation", "gen_mwh", "mwh"]:
        if c in df.columns:
            gen_col = c
            break
    if gen_col is None:
        raise ValueError("Input must contain a 'Gen' column (or alias: generation, gen_mwh, mwh).")

    if "Datetime" not in df.columns:
        raise ValueError("Input must contain a 'Datetime' column.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=False, errors="coerce")
    df[gen_col] = pd.to_numeric(df[gen_col], errors="coerce")
    df = df.dropna(subset=["Datetime", gen_col]).sort_values("Datetime").set_index("Datetime")

    # Aggregate duplicates on the same timestamp (rare, but safe)
    s = df.groupby(level=0)[gen_col].sum()
    # Coerce to exact hourly timeline; lower-case 'h' avoids pandas' deprecation warning
    full_idx = pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h"), freq="h")
    s = s.reindex(full_idx).ffill().bfill()

    s.name = "Gen"
    return s


# -----------------------
# Build clear-sky seasonal baseline
# -----------------------
def build_clear_sky_baseline(gen: pd.Series, doy_window: int = 15, quantile: float = 0.98) -> pd.Series:
    """
    Clear-sky (sunny) baseline:
      1) High-quantile envelope (e.g., 98th pct) by (day-of-year, hour)
      2) Smooth across days-of-year via centered rolling window
      3) Map back to each hourly stamp
    """
    idx = gen.index
    df = pd.DataFrame(
        {"Gen": gen.to_numpy(), "doy": idx.dayofyear, "hour": idx.hour}
    )
    q = (
        df.groupby(["doy", "hour"])["Gen"]
        .quantile(quantile)
        .unstack("hour")
        .sort_index()
    )

    # Ensure full 1..366 day coverage
    all_days = pd.RangeIndex(1, 367, name="doy")
    q = q.reindex(all_days)

    # Smooth across day-of-year with wrap-around padding
    pad = max(1, doy_window // 2)
    q_pad = pd.concat([q.iloc[-pad:], q, q.iloc[:pad]], axis=0)
    q_smooth = q_pad.rolling(window=doy_window, center=True, min_periods=max(3, pad)).mean()
    q_smooth = q_smooth.iloc[pad:-pad].ffill().bfill()

    # Broadcast to hourly history
    baseline_vals = q_smooth.to_numpy()[idx.dayofyear - 1, idx.hour]
    baseline = pd.Series(baseline_vals, index=idx, name="Baseline")

    # Guard against zeros
    eps = max(1e-6, float(np.nanpercentile(baseline_vals, 0.1)) * 1e-6)
    return baseline.clip(lower=eps)


# -----------------------
# Residuals = actual / baseline
# -----------------------
def compute_residuals(gen: pd.Series, baseline: pd.Series,
                      clip_low: float = 0.0, clip_high: float = 1.3) -> pd.Series:
    r = (gen / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip_low is not None:
        r = r.clip(lower=clip_low)
    if clip_high is not None:
        r = r.clip(upper=clip_high)
    r.name = "Residual"
    return r


# -----------------------
# Month-aware block bootstrap of residuals
# -----------------------
def month_block_bootstrap(
    residuals: pd.Series,
    target_index: pd.DatetimeIndex,
    block_len: int,
    month_band: int,
    n_sims: int,
    seed: int | None = 1234,
) -> np.ndarray:
    """
    Return residual samples shaped (n_hours, n_sims) by block bootstrapping:
      - For each month in the target calendar, choose historical start indices from
        months within ± month_band (cyclic).
      - Copy blocks of length block_len until the target month’s hours are filled.
    Fully vectorized over hours; only loops over months (1..12).
    """
    rng = np.random.default_rng(seed)

    hist_idx = residuals.index
    hist_month = hist_idx.month.values.astype(np.int16)
    res_vals = residuals.values
    n_hist = res_vals.shape[0]

    # Precompute pools of valid historical start positions for each calendar month
    month_to_starts: dict[int, np.ndarray] = {}
    for m in range(1, 13):
        allowed = [((m - 1 + d) % 12) + 1 for d in range(-month_band, month_band + 1)]
        mask = np.isin(hist_month, allowed)
        start_idxs = np.where(mask)[0]
        if block_len > 1:
            start_idxs = start_idxs[start_idxs <= (n_hist - block_len)]
        if start_idxs.size == 0:
            # Fallback: anywhere in history that permits a full block
            start_idxs = np.arange(max(1, n_hist - block_len))
        month_to_starts[m] = start_idxs

    n_hours = len(target_index)
    out = np.empty((n_hours, n_sims), dtype=np.float32)

    tgt_month = target_index.month.values
    for m in range(1, 13):
        pos = np.where(tgt_month == m)[0]
        if pos.size == 0:
            continue

        starts_pool = month_to_starts[m]
        n_blocks = math.ceil(pos.size / block_len)

        # Draw shape (n_blocks, n_sims) of starting positions
        draw = rng.choice(starts_pool, size=(n_blocks, n_sims), replace=True)

        # Offsets need shape (block_len, 1, 1) to broadcast with (1, n_blocks, n_sims)
        offs = np.arange(block_len, dtype=np.int64).reshape(block_len, 1, 1)
        idxs = (draw.reshape(1, n_blocks, n_sims) + offs).reshape(block_len * n_blocks, n_sims)

        # Trim to exact month length and gather
        idxs = idxs[: pos.size, :]
        sampled = res_vals[idxs]
        out[pos, :] = sampled.astype(np.float32)

    return out


# -----------------------
# Peak/off-peak labeling
# -----------------------
def label_peak_offpeak(idx: pd.DatetimeIndex, mode: str = "caiso_like") -> np.ndarray:
    """
    True = PEAK, False = OFFPEAK.
    - 'caiso_like' (default): 07:00–22:00 AND Mon–Sat
    - 'simple': 07:00–22:00 every day
    """
    hours = idx.hour.values
    in_peak_hours = (hours >= 7) & (hours <= 22)
    if mode == "simple":
        return in_peak_hours
    days = idx.dayofweek.values  # Mon=0..Sun=6
    return in_peak_hours & (days <= 5)


# -----------------------
# Future timeline + per-year scaling map
# -----------------------
def make_future_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    end = pd.Timestamp(f"{end_year}-12-31 23:00:00")
    return pd.date_range(start, end, freq="h")


def parse_per_year_mult(s: str | None, start_year: int, end_year: int) -> dict[int, float]:
    out = {y: 1.0 for y in range(start_year, end_year + 1)}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        y_str, v_str = part.split(":")
        y = int(y_str.strip())
        v = float(v_str.strip())
        if y in out:
            out[y] = v
    return out


# -----------------------
# Monthly PEAK/OFFPEAK energy → quantiles
# -----------------------
def monthly_quantiles_by_period(
    idx: pd.DatetimeIndex,
    sims: np.ndarray,              # (H, n_sims) MW
    peak_mask: np.ndarray,         # (H,) bool
    q_levels: tuple[float, ...] = (0.5, 0.75, 0.9),
) -> pd.DataFrame:
    """
    Sum hourly MW into monthly MWh for PEAK and OFFPEAK, then compute quantiles across simulations.
    Vectorized with np.bincount; no Python loops over hours.
    """
    years = idx.year.values.astype(np.int32)
    months = idx.month.values.astype(np.int32)

    # Collapse (year, month) to single group id like YYYYMM, then factorize to 0..G-1
    ym_key = years * 100 + months
    group_keys, group_inv = np.unique(ym_key, return_inverse=True)  # group_keys length = G
    group_to_year = (group_keys // 100).astype(int)
    group_to_month = (group_keys % 100).astype(int)
    G = group_keys.size
    n_sims = sims.shape[1]

    out_frames = []
    for period_name, mask in [("PEAK", peak_mask), ("OFFPEAK", ~peak_mask)]:
        sims_masked = np.where(mask[:, None], sims, 0.0)

        # Sum per (year, month) with bincount for each simulation
        monthly_by_sim = np.zeros((G, n_sims), dtype=np.float64)
        for j in range(n_sims):
            monthly_by_sim[:, j] = np.bincount(group_inv, weights=sims_masked[:, j], minlength=G)

        # Quantiles across simulations (axis=1 → per group)
        q_arr = np.quantile(monthly_by_sim, q=q_levels, axis=1)  # shape: (len(q_levels), G)

        # Build one tidy frame with PXX columns
        data = {
            "year": group_to_year,
            "month": group_to_month,
            "period": period_name,
        }
        for qi, qv in zip(q_levels, q_arr):
            data[f"P{int(round(qi*100)):02d}"] = qv
        out_frames.append(pd.DataFrame(data))

    wide = pd.concat(out_frames, ignore_index=True).sort_values(["year", "month", "period"])
    # Column order
    cols = ["year", "month", "period"] + [c for c in ["P50", "P75", "P90"] if c in wide.columns]
    return wide[cols]


# -----------------------
# Main simulate() pipeline
# -----------------------
def simulate(
    input_csv: str | Path,
    start_year: int,
    end_year: int,
    per_year_mult_str: str | None,
    block_len: int,
    month_band: int,
    n_sims: int,
    jitter_frac: float,
    peak_mode: str,
    save_hourly: bool,
    out_dir: str | Path,
    seed: int | None = 1234,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load history & build residuals
    hist = load_gen(input_csv)  # MW
    baseline_hist = build_clear_sky_baseline(hist, doy_window=15, quantile=0.98)
    residuals = compute_residuals(hist, baseline_hist, clip_low=0.0, clip_high=1.3)

    # 2) Future baseline by (doy, hour) using the baseline's own seasonal envelope (median)
    future_idx = make_future_index(start_year, end_year)
    b_df = pd.DataFrame(
        {"b": baseline_hist.values, "doy": hist.index.dayofyear, "hour": hist.index.hour}
    )
    b_env = b_df.groupby(["doy", "hour"])["b"].median().unstack("hour")
    all_days = pd.RangeIndex(1, 367, name="doy")
    b_env = b_env.reindex(all_days).ffill().bfill()
    future_baseline_vals = b_env.to_numpy()[future_idx.dayofyear - 1, future_idx.hour]
    future_baseline = pd.Series(future_baseline_vals, index=future_idx, name="BaselineFuture")

    # 3) Month-aware block bootstrap of residuals into future horizon
    resid_mat = month_block_bootstrap(
        residuals=residuals,
        target_index=future_idx,
        block_len=block_len,
        month_band=month_band,
        n_sims=n_sims,
        seed=seed,
    )  # (H, n_sims)

    # 4) Apply baseline * residuals + optional jitter on high-baseline hours
    sims = future_baseline.values.astype(np.float64)[:, None] * resid_mat.astype(np.float64)
    if jitter_frac > 0.0:
        b = future_baseline.values
        hi_mask = b > np.nanmedian(b)
        rng = np.random.default_rng(None if seed is None else seed + 7)
        noise = rng.normal(loc=0.0, scale=1.0, size=sims.shape) * (jitter_frac * b[:, None])
        sims = np.where(hi_mask[:, None], sims + noise, sims)
        sims = np.clip(sims, a_min=0.0, a_max=None)

    # 5) Per-year scaling
    per_year_mult = parse_per_year_mult(per_year_mult_str, start_year, end_year)
    scale_vec = np.vectorize(per_year_mult.get)(future_idx.year).astype(np.float64)
    sims *= scale_vec[:, None]

    # 6) Monthly PEAK/OFFPEAK quantiles (MWh)
    peak_mask = label_peak_offpeak(future_idx, mode=peak_mode)
    monthly_q = monthly_quantiles_by_period(future_idx, sims, peak_mask, q_levels=(0.5, 0.75, 0.9))
    monthly_q_path = out_dir / f"monthly_quantiles_{start_year}_{end_year}.csv"
    monthly_q.to_csv(monthly_q_path, index=False)

    # 7) Optional hourly save (chunked to keep files manageable)
    if save_hourly:
        try:
            import pyarrow  # noqa: F401
            use_parquet = False
        except Exception:
            use_parquet = False

        base_cols = pd.DataFrame({"Datetime": future_idx})
        chunk = 100
        for s0 in range(0, n_sims, chunk):
            s1 = min(s0 + chunk, n_sims)
            block = sims[:, s0:s1]
            cols = {f"sim_{i}": block[:, i - s0] for i in range(s0, s1)}
            block_df = pd.concat([base_cols, pd.DataFrame(cols)], axis=1)
            if use_parquet:
                p = out_dir / f"hourly_sims_{start_year}_{end_year}_{s0:04d}_{s1-1:04d}.parquet"
                block_df.to_parquet(p, index=False)
            else:
                p = out_dir / f"hourly_sims_{start_year}_{end_year}_{s0:04d}_{s1-1:04d}.csv"
                block_df.to_csv(p, index=False)

    # 8) Save run metadata
    meta = {
        "input_csv": str(input_csv),
        "start_year": start_year,
        "end_year": end_year,
        "per_year_mult": per_year_mult,
        "block_len": block_len,
        "month_band": month_band,
        "n_sims": n_sims,
        "jitter_frac": jitter_frac,
        "peak_mode": peak_mode,
        "save_hourly": save_hourly,
        "seed": seed,
    }
    with open(out_dir / "simulate_caiso_solar_settings.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved monthly quantiles → {monthly_q_path}")
    if save_hourly:
        print(f"[OK] Hourly simulation files saved under: {out_dir}")


# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(
        description="Simulate future CAISO solar via clear-sky baseline + month-aware block-bootstrap residuals."
    )
    p.add_argument("input_csv", type=str, help="CSV with ['Datetime','Gen'] (hourly).")
    p.add_argument("--start_year", type=int, required=True)
    p.add_argument("--end_year", type=int, required=True)
    p.add_argument(
        "--per_year_mult",
        type=str,
        default=None,
        help='Optional per-year scale factors, e.g. "2026:1.00,2027:1.02,2028:1.04". Missing years default to 1.0.',
    )
    p.add_argument("--block_len", type=int, default=12, help="Residual block length in hours (default 12).")
    p.add_argument(
        "--month_band",
        type=int,
        default=1,
        help="Allowed historical months are target_month ± month_band (cyclic).",
    )
    p.add_argument("--n_sims", type=int, default=1000, help="Number of Monte Carlo simulations.")
    p.add_argument(
        "--jitter_frac",
        type=float,
        default=0.01,
        help="Gaussian jitter as a fraction of baseline on high-baseline hours (e.g., 0.01).",
    )
    p.add_argument(
        "--peak_mode",
        type=str,
        choices=["caiso_like", "simple"],
        default="caiso_like",
        help="Peak/Off-peak definition (default: caiso_like).",
    )
    p.add_argument("--save_hourly", action="store_true", help="Also write hourly simulation files (large).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")

    args = p.parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start_year must be <= end_year")

    simulate(
        input_csv=args.input_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        per_year_mult_str=args.per_year_mult,
        block_len=args.block_len,
        month_band=args.month_band,
        n_sims=args.n_sims,
        jitter_frac=args.jitter_frac,
        peak_mode=args.peak_mode,
        save_hourly=args.save_hourly,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
