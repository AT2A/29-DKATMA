# scripts/CAISO_sim_framework.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import math

# ============================================
# CAISO PRICE SIMULATION (same as MISO/ERCOT)
# ============================================
def run_caiso_price_sim(
    cleaned_dir: str | Path = "data/cleaned",
    seed: int | None = None,
    save: bool = False,
    output_dir: str | Path = "data/simulations",
) -> pd.DataFrame:
    """Run a single Monte Carlo simulation projecting CAISO Busbar prices."""
    rng = np.random.default_rng(seed)
    cleaned_dir = Path(cleaned_dir)
    cleaned_price_path = cleaned_dir / "CAISO_cleaned.csv"
    forwards_path = cleaned_dir / "CAISO_forwards.csv"

    if not cleaned_price_path.exists() or not forwards_path.exists():
        raise FileNotFoundError("Missing CAISO cleaned or forwards CSV.")

    print(f"[CAISO_price] reading: {cleaned_price_path}")
    print(f"[CAISO_price] reading: {forwards_path}")

    df = pd.read_csv(cleaned_price_path, parse_dates=["Datetime"])
    fwd = pd.read_csv(forwards_path, parse_dates=["Month"])

    if {"Busbar", "Hub"}.issubset(df.columns):
        df["Real_Spread"] = df["Busbar"] - df["Hub"]
    elif {"RT Busbar", "RT Hub"}.issubset(df.columns):
        df["Real_Spread"] = df["RT Busbar"] - df["RT Hub"]
    else:
        raise ValueError("Could not find valid Hub/Busbar columns for CAISO.")

    if {"DA Busbar", "DA Hub"}.issubset(df.columns):
        df["DA_Spread"] = df["DA Busbar"] - df["DA Hub"]
    else:
        df["DA_Spread"] = df["Real_Spread"]

    df["Avg_Spread"] = (df["Real_Spread"] + df["DA_Spread"]) / 2
    df["Month"] = df["Datetime"].dt.month

    monthly = (
        df.groupby("Month")["Avg_Spread"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "Spread_Mean", "std": "Spread_Std"})
        .reset_index()
    )

    fwd["Month"] = fwd["Month"].dt.month
    proj = fwd.merge(monthly, on="Month", how="left")
    proj["Noise"] = rng.normal(0, proj["Spread_Std"])
    proj["Busbar_Projected"] = proj["Forward_Peak"] + proj["Spread_Mean"] + proj["Noise"]

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "CAISO_Busbar_Simulation_single.csv"
        proj.to_csv(out, index=False)
        print(f"✅ Saved single Busbar simulation to {out.resolve()}")

    return proj


# ============================================
# CAISO GENERATION SIMULATION (solar logic)
# ============================================
def run_caiso_gen_sim(
    input_csv: str | Path | None = None,
    hist_series: pd.Series | None = None,
    start_year: int = 2026,
    end_year: int = 2030,
    seed: int | None = None,
    block_len: int = 12,
    month_band: int = 1,
    cagr: float | None = None,
    base_mult: float = 1.0,
    save: bool = False,
    output_dir: str | Path = "data/simulations",
) -> pd.Series:
    """
    Run a synthetic CAISO solar generation simulation:
      - Builds a clear-sky baseline from historical data
      - Computes residuals (actual / baseline)
      - Bootstraps residuals month-aware
      - Recombines with baseline for future years
      - Applies CAGR scaling
    """
    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1. Load or accept historical series
    # -----------------------------
    if hist_series is None:
        if input_csv is None:
            raise ValueError("Must provide either hist_series or input_csv.")
        input_csv = Path(input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"Missing file: {input_csv}")
        print(f"[CAISO_gen] reading: {input_csv}")
        df = pd.read_csv(input_csv, parse_dates=["Datetime"])
        hist_series = pd.Series(pd.to_numeric(df["Gen"], errors="coerce").values,
                                index=df["Datetime"]).dropna()

    gen = hist_series.sort_index()

    # -----------------------------
    # 2. Build clear-sky baseline
    # -----------------------------
    idx = gen.index
    df = pd.DataFrame({"Gen": gen.values, "doy": idx.dayofyear, "hour": idx.hour})
    q = df.groupby(["doy", "hour"])["Gen"].quantile(0.98).unstack("hour").sort_index()
    all_days = pd.RangeIndex(1, 367, name="doy")
    q = q.reindex(all_days).interpolate(limit_direction="both")
    q_smooth = q.rolling(window=15, center=True, min_periods=3).mean().ffill().bfill()
    baseline_vals = q_smooth.to_numpy()[idx.dayofyear - 1, idx.hour]
    baseline = pd.Series(baseline_vals, index=idx, name="Baseline")
    baseline = baseline.clip(lower=1e-3)

    # -----------------------------
    # 3. Compute residuals
    # -----------------------------
    resid = (gen / baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    resid = resid.clip(lower=0.0, upper=1.3)

    # -----------------------------
    # 4. Month-aware block bootstrap of residuals
    # -----------------------------
    hist_month = resid.index.month.values
    res_vals = resid.values
    n_hist = len(res_vals)

    def sample_month_block(target_month: int) -> np.ndarray:
        allowed = [((target_month - 1 + d) % 12) + 1 for d in range(-month_band, month_band + 1)]
        mask = np.isin(hist_month, allowed)
        start_idxs = np.where(mask)[0]
        if block_len > 1:
            start_idxs = start_idxs[start_idxs <= (n_hist - block_len)]
        if start_idxs.size == 0:
            start_idxs = np.arange(max(1, n_hist - block_len))
        start = rng.choice(start_idxs)
        return res_vals[start:start + block_len]

    # -----------------------------
    # 5. Generate future baseline
    # -----------------------------
    future_idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31 23:00", freq="h")
    b_df = pd.DataFrame({"b": baseline.values, "doy": idx.dayofyear, "hour": idx.hour})
    b_env = b_df.groupby(["doy", "hour"])["b"].median().unstack("hour")
    b_env = b_env.reindex(all_days).ffill().bfill()
    future_baseline_vals = b_env.to_numpy()[future_idx.dayofyear - 1, future_idx.hour]
    future_baseline = pd.Series(future_baseline_vals, index=future_idx)

    # -----------------------------
    # 6. Bootstrap residuals into future timeline
    # -----------------------------
    vals, pos = [], 0
    while pos < len(future_idx):
        m = future_idx[pos].month
        block = sample_month_block(m)
        take = min(len(block), len(future_idx) - pos)
        vals.extend(block[:take])
        pos += take
    resid_future = np.asarray(vals, dtype=float)

    # -----------------------------
    # 7. Recombine baseline × residuals
    # -----------------------------
    sim_vals = future_baseline.values * resid_future

    # -----------------------------
    # 8. Apply CAGR scaling
    # -----------------------------
    years = list(range(start_year, end_year + 1))
    y0 = years[0]
    for y in years:
        factor = base_mult * ((1 + (cagr or 0.0)) ** (y - y0))
        sim_vals[future_idx.year == y] *= factor

    sim = pd.Series(sim_vals, index=future_idx, name="Gen")

    # -----------------------------
    # 9. Save optionally
    # -----------------------------
    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"CAISO_Gen_Simulation_{start_year}_{end_year}.csv"
        sim.to_csv(out_path, index_label="Datetime")
        print(f"✅ Saved single Gen simulation to {out_path.resolve()}")

    return sim


if __name__ == "__main__":
    price_df = run_caiso_price_sim(save=False)
    gen_df = run_caiso_gen_sim(save=False)
    print("\n✅ Finished one CAISO price & generation simulation run.")