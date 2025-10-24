from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from time import perf_counter

# Frameworks
from scripts.MISO_sim_framework import run_miso_gen_sim
from scripts.ERCOT_sim_framework import run_ercot_gen_sim

# ----------------------------
# Helpers
# ----------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def month_start_index(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize timestamps to the first day of their month (00:00)."""
    return pd.to_datetime([pd.Timestamp(y, m, 1) for y, m in zip(dt_index.year, dt_index.month)])

def compute_monthly_gen_mwh(gen_series: pd.Series) -> pd.DataFrame:
    """Sum hourly Gen to monthly MWh. Returns a DataFrame indexed by MonthStart."""
    gen_series = gen_series.copy()
    gen_series.index = month_start_index(gen_series.index)
    monthly = gen_series.groupby(gen_series.index).sum()
    return pd.DataFrame({"Gen_MWh": monthly})

def precompute_price_inputs(cleaned_dir: Path, company: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load spread stats and forward curve for MISO or ERCOT."""
    cleaned_dir = Path(cleaned_dir)
    mclean = cleaned_dir / f"{company}_cleaned.csv"
    mforwd = cleaned_dir / f"{company}_forwards.csv"
    if not mclean.exists():
        raise FileNotFoundError(f"Missing file: {mclean}")
    if not mforwd.exists():
        raise FileNotFoundError(f"Missing file: {mforwd}")

    df  = pd.read_csv(mclean, parse_dates=["Datetime"])
    fwd = pd.read_csv(mforwd, parse_dates=["Month"])

    # Spread stats by calendar month (1..12)
    if company == "MISO":
        df["Real_Spread"] = df["Busbar"] - df["Hub"]
        df["DA_Spread"]   = df["DA Busbar"] - df["DA Hub"]
    elif company == "ERCOT":
        # Accept RT columns as fallback
        busbar = df["Busbar"] if "Busbar" in df.columns else df["RT Busbar"]
        hub    = df["Hub"]    if "Hub"    in df.columns else df["RT Hub"]
        df["Real_Spread"] = busbar - hub
        if {"DA Busbar", "DA Hub"}.issubset(df.columns):
            df["DA_Spread"] = df["DA Busbar"] - df["DA Hub"]
        else:
            df["DA_Spread"] = df["Real_Spread"]
    else:
        raise NotImplementedError(f"Company '{company}' not supported.")

    df["Avg_Spread"] = (df["Real_Spread"] + df["DA_Spread"]) / 2
    df["MonthNum"]   = df["Datetime"].dt.month
    spread_stats = (
        df.groupby("MonthNum")["Avg_Spread"]
          .agg(["mean", "std"])
          .rename(columns={"mean": "Spread_Mean", "std": "Spread_Std"})
          .reset_index()
    )

    # Forward months as proper month keys
    fwd["MonthStart"] = pd.to_datetime(fwd["Month"].dt.to_period("M").dt.start_time)
    fwd["MonthNum"]   = fwd["MonthStart"].dt.month
    return spread_stats, fwd

def simulate_busbar_matrix(spread_stats: pd.DataFrame, fwd: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """
    Vectorized price simulation:
      For each forward month j, Busbar[:, j] = Forward_Peak_j + N(Spread_Mean_m, Spread_Std_m)
      where m = calendar month number of that j.
    Returns a DataFrame with index MonthStart and columns Sim1..SimN.
    """
    rng = np.random.default_rng()  # fresh randomness each run
    fwd = fwd.merge(spread_stats, on="MonthNum", how="left")
    mu  = (fwd["Forward_Peak"] + fwd["Spread_Mean"]).to_numpy(dtype=float)  # (num_months,)
    sig = fwd["Spread_Std"].to_numpy(dtype=float)                            # (num_months,)

    noise   = rng.normal(loc=0.0, scale=sig[:, None], size=(len(fwd), n_sims))
    busbar  = mu[:, None] + noise  # (num_months, n_sims)

    cols = [f"Sim{i+1}" for i in range(n_sims)]
    return pd.DataFrame(busbar, index=fwd["MonthStart"], columns=cols)

def fmt_secs(s: float) -> str:
    """Pretty print seconds as H:MM:SS.mmm."""
    h = int(s // 3600); s -= h*3600
    m = int(s // 60);   s -= m*60
    return f"{h:d}:{m:02d}:{s:06.3f}"

# ----------------------------
# Main Monte Carlo Runner
# ----------------------------
def run_revenue_simulation(company: str, n_sims: int, reuse_gen: bool = False, seed: int = 42, timings: bool = False):
    """
    Monte Carlo revenue simulation for MISO/ERCOT.
    Default: reuse_gen=False → new Gen simulation per run (full Monte Carlo).
    """
    company = company.upper()
    if company not in {"MISO", "ERCOT"}:
        raise NotImplementedError(f"Company '{company}' not supported. Use MISO or ERCOT.")

    ROOT        = project_root()
    CLEANED_DIR = ROOT / "data" / "cleaned"

    print(f"\n🚀 Running {n_sims} Monte Carlo revenue simulations for {company} (reuse_gen={reuse_gen})\n")

    t0 = perf_counter()

    # 1) Price inputs once
    t_price0 = perf_counter()
    spread_stats, fwd = precompute_price_inputs(CLEANED_DIR, company)
    t_price1 = perf_counter()

    # 2) Load historical generation ONCE (in-memory)
    #    This avoids re-reading CSV on each simulation.
    t_gen0 = perf_counter()
    gen_clean_path = CLEANED_DIR / f"{company}_cleaned.csv"
    base_hist_df = pd.read_csv(gen_clean_path, parse_dates=["Datetime"])
    base_hist_series = pd.Series(pd.to_numeric(base_hist_df["Gen"], errors="coerce").values,
                                 index=base_hist_df["Datetime"]).dropna()
    # If user chose reuse_gen=True, build one synthetic path now
    if reuse_gen:
        if company == "MISO":
            gen_series = run_miso_gen_sim(hist_series=base_hist_series, save=False)
        else:
            gen_series = run_ercot_gen_sim(hist_series=base_hist_series, save=False)
        monthly_gen = compute_monthly_gen_mwh(gen_series)
    else:
        monthly_gen = None
    t_gen1 = perf_counter()

    # 3) Vectorized simulate busbar for all months × sims
    t_vec0 = perf_counter()
    busbar_mat = simulate_busbar_matrix(spread_stats, fwd, n_sims=n_sims)
    if reuse_gen:
        monthly_gen = monthly_gen.reindex(busbar_mat.index).ffill().fillna(0.0)
    t_vec1 = perf_counter()

    # 4) Monte Carlo loop: compute revenue for each simulation
    revenues = []
    total_gen_each_sim = []  # <-- added initialization
    per_sim_times = []

    for i in range(n_sims):
        ts0 = perf_counter()

        # --- Generation ---
        if reuse_gen:
            gen_mwh = monthly_gen["Gen_MWh"].to_numpy(dtype=float)
        else:
            # New Gen path each sim (in-memory base history)
            if company == "MISO":
                gen_series_i = run_miso_gen_sim(hist_series=base_hist_series, seed=seed + i, save=False)
            else:
                gen_series_i = run_ercot_gen_sim(hist_series=base_hist_series, seed=seed + i, save=False)
            mgen_i = compute_monthly_gen_mwh(gen_series_i).reindex(busbar_mat.index).ffill().fillna(0.0)
            gen_mwh = mgen_i["Gen_MWh"].to_numpy(dtype=float)

        total_gen_each_sim.append(float(np.sum(gen_mwh)))  # <-- record total MWh

        # --- Prices & Revenue ---
        prices_i = busbar_mat.iloc[:, i].to_numpy(dtype=float)
        revenues.append(float(np.dot(gen_mwh, prices_i)))

        ts1 = perf_counter()
        per_sim_times.append(ts1 - ts0)
        if timings:
            print(f"sim {i+1}/{n_sims} completed in {fmt_secs(ts1 - ts0)}")

    # --- summarize once, after loop ---
    t1 = perf_counter()
    total_time = t1 - t0
    avg_time_per_sim = (sum(per_sim_times) / n_sims) if n_sims > 0 else 0.0

    revenues = np.array(revenues, dtype=float)
    mean_rev, std_rev = revenues.mean(), revenues.std()
    p75_rev = np.percentile(revenues, 75)

    # Determine denominator for fixed price
    total_gen_each_sim = np.array(total_gen_each_sim, dtype=float)
    if reuse_gen:
        denom_mwh = float(total_gen_each_sim[0]) if len(total_gen_each_sim) > 0 else np.nan
    else:
        denom_mwh = float(total_gen_each_sim.mean()) if len(total_gen_each_sim) > 0 else np.nan

    fixed_price_mean = mean_rev / denom_mwh if denom_mwh and denom_mwh > 0 else np.nan
    fixed_price_p75  = p75_rev  / denom_mwh if denom_mwh and denom_mwh > 0 else np.nan

    print("\nrevenue distribution summary")
    print(f"mean: ${mean_rev:,.2f}")
    print(f"std dev: ${std_rev:,.2f}")
    print(f"75th percentile: ${p75_rev:,.2f}")

    print("\nfixed price (using avg total MWh denominator)")
    if not np.isnan(fixed_price_mean):
        print(f"at mean revenue: ${fixed_price_mean:,.2f}/MWh")
    if not np.isnan(fixed_price_p75):
        print(f"at 75th pct rev: ${fixed_price_p75:,.2f}/MWh")

    print(f"\n⏱️ timing")
    print(f"price prep:     {fmt_secs(t_price1 - t_price0)}")
    print(f"gen prep:       {fmt_secs(t_gen1 - t_gen0)}")
    print(f"vectorized px:  {fmt_secs(t_vec1 - t_vec0)}")
    print(f"per-sim (avg):  {fmt_secs(avg_time_per_sim)}")
    print(f"total runtime:  {fmt_secs(total_time)}\n")

    return revenues, p75_rev

# ----------------------------
# CLI entry point
# ----------------------------
if __name__ == "__main__":
    # Usage: python -m scripts.Monte_Carlo_sim MISO 100 [--reuse-gen] [--timings]
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.Monte_Carlo_sim <company> <n_sims> [--reuse-gen] [--timings]")
        print("       (default uses a new Gen path for each simulation)")
        sys.exit(1)

    company = sys.argv[1]
    n_sims = int(sys.argv[2])
    reuse = False   # default: new Gen each sim
    timings = False

    for arg in sys.argv[3:]:
        if arg == "--reuse-gen":
            reuse = True
        elif arg == "--timings":
            timings = True

    run_revenue_simulation(company, n_sims, reuse_gen=reuse, seed=42, timings=timings)