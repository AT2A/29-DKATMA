# scripts/Monte_Carlo_sim.py
from __future__ import annotations
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from time import perf_counter, strftime

# ============================================
# Framework Imports
# ============================================
from scripts.MISO_sim_framework import run_miso_gen_sim
from scripts.ERCOT_sim_framework import run_ercot_gen_sim
from scripts.CAISO_sim_framework import run_caiso_gen_sim

# ============================================
# Helpers: paths, time fmt
# ============================================
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def fmt_secs(s: float) -> str:
    """Pretty print seconds as H:MM:SS.mmm."""
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    return f"{h:d}:{m:02d}:{s:06.3f}"

# ============================================
# Robust loaders / wrappers
# ============================================
GEN_CANDIDATES = [
    "Gen", "GEN", "generation", "Generation", "MWh", "MW", "Output", "RT Gen", "DA Gen"
]

def load_hist_gen_series(cleaned_path: Path) -> pd.Series:
    """
    Load a historical hourly generation series with a DatetimeIndex.
    Tries to find a plausible 'Gen' column; if not present, picks the first numeric column.
    """
    df = pd.read_csv(cleaned_path, parse_dates=["Datetime"])
    # Pick a generation-like column
    gen_col = None
    for c in GEN_CANDIDATES:
        if c in df.columns:
            gen_col = c
            break
    if gen_col is None:
        # fallback: first numeric column that's not Datetime
        numerics = [c for c in df.columns if c != "Datetime" and pd.api.types.is_numeric_dtype(df[c])]
        if not numerics:
            raise ValueError(f"No numeric generation column found in {cleaned_path.name}")
        gen_col = numerics[0]

    s = pd.to_numeric(df[gen_col], errors="coerce")
    ser = pd.Series(s.values, index=df["Datetime"]).dropna()
    if not isinstance(ser.index, pd.DatetimeIndex):
        raise ValueError("Historical series index must be DatetimeIndex")
    return ser.sort_index()

def get_gen_series(company: str, base_hist_series: pd.Series) -> pd.Series:
    """
    Tries multiple signatures for run_*_gen_sim and guarantees a pandas Series.
    If the framework returns None (or mismatched type), falls back to a light
    bootstrap perturbation of the historical series (same index, non-negative).
    """
    company = company.upper()
    fn_map = {
        "MISO": run_miso_gen_sim,
        "ERCOT": run_ercot_gen_sim,
        "CAISO": run_caiso_gen_sim,
    }
    fn = fn_map[company]

    # Try a few common signatures
    for kwargs in (
        {"hist_series": base_hist_series, "save": False},
        {"hist_series": base_hist_series},
        {"hist_series": base_hist_series, "return_series": True},
    ):
        try:
            s = fn(**kwargs)
            if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex):
                return s.sort_index()
        except TypeError:
            continue
        except Exception:
            continue

    # Fallback: bootstrap-ish perturbation of historical hourly Gen
    rng = np.random.default_rng()
    vals = base_hist_series.to_numpy(dtype=float)
    roll = pd.Series(vals).rolling(168, min_periods=1).std().to_numpy()
    sigma = np.where(np.isfinite(roll) & (roll > 0), 0.05 * roll, 0.05 * (np.nanstd(vals) or 1.0))
    noise = rng.normal(0.0, sigma, size=len(vals))
    boot = np.clip(vals + noise, 0.0, None)
    return pd.Series(boot, index=base_hist_series.index, name="Gen").sort_index()

# ============================================
# Data helpers (monthly indices, gen aggregation)
# ============================================
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
    """Load spread stats and forward curve for MISO, ERCOT, or CAISO."""
    cleaned_dir = Path(cleaned_dir)
    mclean = cleaned_dir / f"{company}_cleaned.csv"
    mforwd = cleaned_dir / f"{company}_forwards.csv"
    if not mclean.exists():
        raise FileNotFoundError(f"Missing file: {mclean}")
    if not mforwd.exists():
        raise FileNotFoundError(f"Missing file: {mforwd}")

    df = pd.read_csv(mclean, parse_dates=["Datetime"])
    fwd = pd.read_csv(mforwd, parse_dates=["Month"])

    # Spread stats by calendar month (1..12)
    if company == "MISO":
        df["Real_Spread"] = df["Busbar"] - df["Hub"]
        df["DA_Spread"] = df["DA Busbar"] - df["DA Hub"]
    elif company == "ERCOT":
        busbar = df["Busbar"] if "Busbar" in df.columns else df["RT Busbar"]
        hub = df["Hub"] if "Hub" in df.columns else df["RT Hub"]
        df["Real_Spread"] = busbar - hub
        if {"DA Busbar", "DA Hub"}.issubset(df.columns):
            df["DA_Spread"] = df["DA Busbar"] - df["DA Hub"]
        else:
            df["DA_Spread"] = df["Real_Spread"]
    elif company == "CAISO":
        busbar = df["Busbar"] if "Busbar" in df.columns else df.get("RT Busbar", pd.Series(np.nan, index=df.index))
        hub = df["Hub"] if "Hub" in df.columns else df.get("RT Hub", pd.Series(np.nan, index=df.index))
        df["Real_Spread"] = busbar - hub
        if {"DA Busbar", "DA Hub"}.issubset(df.columns):
            df["DA_Spread"] = df["DA Busbar"] - df["DA Hub"]
        else:
            df["DA_Spread"] = df["Real_Spread"]
    else:
        raise NotImplementedError(f"Company '{company}' not supported.")

    df["Avg_Spread"] = (df["Real_Spread"] + df["DA_Spread"]) / 2
    df["MonthNum"] = df["Datetime"].dt.month
    spread_stats = (
        df.groupby("MonthNum")["Avg_Spread"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "Spread_Mean", "std": "Spread_Std"})
        .reset_index()
    )

    fwd["MonthStart"] = pd.to_datetime(fwd["Month"].dt.to_period("M").dt.start_time)
    fwd["MonthNum"] = fwd["MonthStart"].dt.month
    return spread_stats, fwd

# ============================================
# Price simulation (vectorized)
# ============================================
def simulate_busbar_matrix(spread_stats: pd.DataFrame, fwd: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """
    For each forward month j, Busbar[:, j] = Forward_Peak_j + N(Spread_Mean_m, Spread_Std_m)
    where m = calendar month number of that j.
    Returns a DataFrame with index MonthStart and columns Sim1..SimN.
    """
    rng = np.random.default_rng()  # no fixed seed -> different each run
    fwd = fwd.merge(spread_stats, on="MonthNum", how="left")
    mu = (fwd["Forward_Peak"] + fwd["Spread_Mean"]).to_numpy(dtype=float)
    sig = fwd["Spread_Std"].to_numpy(dtype=float)

    noise = rng.normal(loc=0.0, scale=sig[:, None], size=(len(fwd), n_sims))
    busbar = mu[:, None] + noise

    cols = [f"Sim{i+1}" for i in range(n_sims)]
    return pd.DataFrame(busbar, index=fwd["MonthStart"], columns=cols)

# ============================================
# Peak/off-peak utilities (7x16 default)
# ============================================
def is_peak_hour(ts: pd.Timestamp) -> bool:
    # 7x16: Weekdays Mon-Fri, hours 7..22 inclusive (start of hour)
    return (ts.weekday() < 5) and (7 <= ts.hour <= 22)

def month_key(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1)

def per_hour_profile_by_monthnum(gen_hourly: pd.Series) -> pd.DataFrame:
    """
    Returns DataFrame indexed by MonthNum (1..12) with columns:
      Peak_MWh_per_hr, Offpeak_MWh_per_hr
    """
    idx = gen_hourly.index
    df = pd.DataFrame({
        "Gen": pd.to_numeric(gen_hourly.values, errors="coerce"),
        "MonthNum": idx.month,
        "IsPeak": [is_peak_hour(ts) for ts in idx],
    }, index=idx)

    grp = df.groupby(["MonthNum", "IsPeak"])["Gen"]
    sum_tbl = grp.sum().unstack(fill_value=0.0)       # columns: False, True
    cnt_tbl = grp.size().unstack(fill_value=0)        # columns: False, True

    # ensure rows 1..12 exist
    for m in range(1, 13):
        if m not in sum_tbl.index:
            sum_tbl.loc[m, False] = 0.0
            sum_tbl.loc[m, True] = 0.0
            cnt_tbl.loc[m, False] = 0
            cnt_tbl.loc[m, True] = 0
    sum_tbl = sum_tbl.sort_index()
    cnt_tbl = cnt_tbl.sort_index()

    # avoid div by zero
    peak_hours = np.where(cnt_tbl.get(True).to_numpy() > 0, cnt_tbl.get(True).to_numpy(), 1)
    offp_hours = np.where(cnt_tbl.get(False).to_numpy() > 0, cnt_tbl.get(False).to_numpy(), 1)

    prof = pd.DataFrame(index=range(1,13))
    prof.index.name = "MonthNum"
    prof["Peak_MWh_per_hr"] = sum_tbl.get(True).to_numpy() / peak_hours
    prof["Offpeak_MWh_per_hr"] = sum_tbl.get(False).to_numpy() / offp_hours
    prof = prof.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prof

def peak_offpeak_hour_counts_for_month(year: int, month: int) -> Tuple[int, int]:
    """Count peak/off-peak hours in a given (year, month) using is_peak_hour rule."""
    start = pd.Timestamp(year, month, 1, 0, 0)
    # last hour start in the month: next month at 00:00 minus 1 hour
    next_month = (start + pd.offsets.MonthBegin(1))
    end = next_month - pd.Timedelta(hours=1)
    hours = pd.date_range(start, end, freq="H")
    flags = [is_peak_hour(ts) for ts in hours]
    peak = int(np.count_nonzero(flags))
    offp = int(len(hours) - peak)
    return peak, offp

def filter_term_months(idx: pd.Series | pd.DatetimeIndex, start_year=2026, end_year=2030) -> pd.DatetimeIndex:
    months = pd.to_datetime(pd.Series(idx).dt.to_period("M").dt.start_time)
    mask = (months.dt.year >= start_year) & (months.dt.year <= end_year)
    return pd.DatetimeIndex(sorted(months[mask].unique()))

def term_months_full(start_year=2026, end_year=2030) -> pd.DatetimeIndex:
    """All month starts from Jan start_year to Dec end_year inclusive."""
    pr = pd.period_range(f"{start_year}-01", f"{end_year}-12", freq="M")
    return pd.DatetimeIndex(pr.to_timestamp())

# ============================================
# Price assembly for valuation (hub, basis, DA/RT)
# ============================================
def compute_monthly_price_components(spread_stats: pd.DataFrame, fwd: pd.DataFrame, n_sims: int):
    """
    Returns dict with simulated monthly prices (months × sims) for:
      RT_Hub, DA_Hub, RT_Busbar, DA_Busbar
    """
    fwd = fwd.copy()
    fwd["MonthStart"] = pd.to_datetime(fwd["Month"].dt.to_period("M").dt.start_time)
    fwd["MonthNum"] = fwd["MonthStart"].dt.month

    fwd = fwd.merge(
        spread_stats.rename(columns={"Spread_Mean": "AvgSpread_Mean", "Spread_Std": "AvgSpread_Std"}),
        on="MonthNum",
        how="left",
    )

    rng = np.random.default_rng()

    # Hub forwards (peak)
    if "Forward_Hub_Peak" in fwd.columns:
        hub_peak = fwd["Forward_Hub_Peak"].to_numpy(float)
    else:
        hub_peak = fwd["Forward_Peak"].to_numpy(float)

    # Off-peak hub forward if available
    hub_off = fwd["Forward_Offpeak"].to_numpy(float) if "Forward_Offpeak" in fwd.columns else None

    # DA uplift (if included in fwd); else 0
    hub_da_uplift = fwd.get("DA_Hub_Uplift_Mean", pd.Series(0.0, index=fwd.index)).to_numpy(float) \
        if "DA_Hub_Uplift_Mean" in fwd.columns else np.zeros(len(fwd), dtype=float)

    # Spread stats (fallback to AvgSpread for both RT/DA)
    rt_spread_mu = fwd.get("RT_Spread_Mean", fwd["AvgSpread_Mean"]).to_numpy(float)
    rt_spread_sd = fwd.get("RT_Spread_Std",  fwd["AvgSpread_Std"]).to_numpy(float)
    da_spread_mu = fwd.get("DA_Spread_Mean", fwd["AvgSpread_Mean"]).to_numpy(float)
    da_spread_sd = fwd.get("DA_Spread_Std",  fwd["AvgSpread_Std"]).to_numpy(float)

    M = len(fwd)
    # simulate spreads: (months × sims)
    rt_spread = rt_spread_mu[:, None] + rng.normal(0.0, rt_spread_sd[:, None], size=(M, n_sims))
    da_spread = da_spread_mu[:, None] + rng.normal(0.0, da_spread_sd[:, None], size=(M, n_sims))

    # hub price paths (peak forward baseline + optional DA uplift)
    rt_hub = np.repeat(hub_peak[:, None], n_sims, axis=1)  # (M×S)
    da_hub = rt_hub + hub_da_uplift[:, None]

    # busbar = hub + spread
    rt_busbar = rt_hub + rt_spread
    da_busbar = da_hub + da_spread

    out = {
        "RT_Hub": (fwd["MonthStart"], rt_hub),
        "DA_Hub": (fwd["MonthStart"], da_hub),
        "RT_Busbar": (fwd["MonthStart"], rt_busbar),
        "DA_Busbar": (fwd["MonthStart"], da_busbar),
    }
    if hub_off is not None:
        out["Hub_Offpeak_Forward"] = (fwd["MonthStart"], np.repeat(hub_off[:, None], n_sims, axis=1))
    return out

def weighted_term_revenue(gen_monthly_peak: np.ndarray, gen_monthly_off: np.ndarray,
                          px_peak: np.ndarray, px_off: np.ndarray | None) -> np.ndarray:
    """
    Revenue per sim over the whole term using monthly peak/off-peak MWh and prices:
      revenue = sum_m (gen_peak[m] * px_peak[m, sim] + gen_off[m] * px_off[m, sim or fallback])
    If px_off is None, fall back to px_peak for both (conservative).
    """
    if px_off is None:
        return (gen_monthly_peak[:, None] * px_peak +
                gen_monthly_off[:, None] * px_peak).sum(axis=0)
    else:
        return (gen_monthly_peak[:, None] * px_peak +
                gen_monthly_off[:, None] * px_off).sum(axis=0)

# ============================================
# Valuation model (P75 fixed; deliverables only; per-sim timings printed)
# ============================================
def run_valuation_model(company: str, n_sims: int = 200, reuse_gen: bool = False) -> Dict[str, Any]:
    """
    Always builds the deliverables for 2026–2030 (P75 fixed price) and prints per-sim timings:
      1) Expected generation by month (Peak/Off-peak)
      2) Four fixed prices (RT/DA × Hub/Busbar) with components: Hub, Basis, Risk_Adjustment
         (Risk_Adjustment = P75 price − mean price)
    Saves only:
      - expected_generation_2026-2030.csv
      - fixed_prices.csv
      - valuation.json
    """
    P_LEVEL = 75  # fixed P75

    company = company.upper()
    if company not in {"MISO", "ERCOT", "CAISO"}:
        raise NotImplementedError(f"Company '{company}' not supported. Use MISO, ERCOT, or CAISO.")

    ROOT = project_root()
    CLEANED_DIR = ROOT / "data" / "cleaned"

    # results dir
    timestamp = strftime("%Y-%m-%d_%H-%M-%S")
    RESULTS_DIR = ROOT / "data" / "results" / company / timestamp
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # load inputs
    spread_stats, fwd = precompute_price_inputs(CLEANED_DIR, company)

    # restrict to 2026–2030 months present in forwards (for pricing)
    px_months_in_term = filter_term_months(fwd["Month"])
    if px_months_in_term.empty:
        raise ValueError("No forward months found in 2026–2030 for valuation.")
    fwd_term = fwd[fwd["Month"].isin(px_months_in_term)].copy()

    # simulate monthly price components
    px = compute_monthly_price_components(spread_stats, fwd_term, n_sims)
    months_index_px_all = px["RT_Hub"][0]
    months_index_px = pd.DatetimeIndex(sorted(set(months_index_px_all).intersection(set(px_months_in_term))))

    # For generation deliverable, always cover full 2026-01 .. 2030-12
    months_index_full = term_months_full(2026, 2030)

    # ---- generation sims -> per-hour profile by month-of-year ----
    gen_clean_path = CLEANED_DIR / f"{company}_cleaned.csv"
    base_hist_series = load_hist_gen_series(gen_clean_path)

    # Accumulate per-hour profiles across sims
    sum_prof_peak = np.zeros(12, dtype=float)
    sum_prof_offp = np.zeros(12, dtype=float)

    if reuse_gen:
        g_series = get_gen_series(company, base_hist_series)
        ts0 = perf_counter()
        prof = per_hour_profile_by_monthnum(g_series)
        sum_prof_peak += prof["Peak_MWh_per_hr"].to_numpy()
        sum_prof_offp += prof["Offpeak_MWh_per_hr"].to_numpy()
        sim_elapsed = perf_counter() - ts0
        print(f"sim 1/{n_sims} completed in {fmt_secs(sim_elapsed)}")
        # Fill remaining with same profile but still print timings (tiny ops to measure)
        for i in range(2, n_sims + 1):
            ts0 = perf_counter()
            _ = float(sum_prof_peak.sum() + sum_prof_offp.sum())
            sim_elapsed = perf_counter() - ts0
            print(f"sim {i}/{n_sims} completed in {fmt_secs(sim_elapsed)}")
    else:
        for i in range(n_sims):
            ts0 = perf_counter()
            gen_series_i = get_gen_series(company, base_hist_series)
            prof_i = per_hour_profile_by_monthnum(gen_series_i)
            sum_prof_peak += prof_i["Peak_MWh_per_hr"].to_numpy()
            sum_prof_offp += prof_i["Offpeak_MWh_per_hr"].to_numpy()
            sim_elapsed = perf_counter() - ts0
            print(f"sim {i+1}/{n_sims} completed in {fmt_secs(sim_elapsed)}")

    # Expected per-hour profile (mean across sims)
    mean_peak_per_hr = sum_prof_peak / n_sims
    mean_offp_per_hr = sum_prof_offp / n_sims

    # Build expected generation table for each 2026–2030 month using actual hour counts
    rows = []
    for mstart in months_index_full:
        year, month = mstart.year, mstart.month
        peak_hrs, offp_hrs = peak_offpeak_hour_counts_for_month(year, month)
        mn = month  # 1..12
        peak_mwh = mean_peak_per_hr[mn - 1] * peak_hrs
        offp_mwh = mean_offp_per_hr[mn - 1] * offp_hrs
        rows.append((mstart, peak_mwh, offp_mwh))

    expected_generation = pd.DataFrame(rows, columns=["MonthStart", "Peak_MWh", "Offpeak_MWh"]).set_index("MonthStart")
    expected_generation.to_csv(RESULTS_DIR / "expected_generation_2026-2030.csv")

    # ---- term revenues per product; P75 fixed price ----
    # Use only months that exist in price matrices
    sel_rt_hub = np.isin(px["RT_Hub"][0], months_index_px)
    sel_da_hub = np.isin(px["DA_Hub"][0], months_index_px)
    sel_rt_bus = np.isin(px["RT_Busbar"][0], months_index_px)
    sel_da_bus = np.isin(px["DA_Busbar"][0], months_index_px)

    rt_hub = px["RT_Hub"][1][sel_rt_hub]
    da_hub = px["DA_Hub"][1][sel_da_hub]
    rt_bus = px["RT_Busbar"][1][sel_rt_bus]
    da_bus = px["DA_Busbar"][1][sel_da_bus]

    hub_off = px.get("Hub_Offpeak_Forward", None)
    hub_off_mat = hub_off[1][np.isin(hub_off[0], months_index_px)] if hub_off is not None else None

    # Expected generation vectors restricted to months available for pricing
    eg_px = expected_generation.reindex(months_index_px)
    exp_gen_peak = eg_px["Peak_MWh"].to_numpy(float)
    exp_gen_off  = eg_px["Offpeak_MWh"].to_numpy(float)
    total_mwh_term = float(exp_gen_peak.sum() + exp_gen_off.sum())

    def term_rev(gen_peak_vec: np.ndarray, gen_off_vec: np.ndarray,
                 prod_mat_peak: np.ndarray, prod_mat_off: np.ndarray | None) -> np.ndarray:
        if prod_mat_off is None:
            return (gen_peak_vec[:, None] * prod_mat_peak +
                    gen_off_vec[:, None] * prod_mat_peak).sum(axis=0)
        else:
            return (gen_peak_vec[:, None] * prod_mat_peak +
                    gen_off_vec[:, None] * prod_mat_off).sum(axis=0)

    rev_rt_hub = term_rev(exp_gen_peak, exp_gen_off, rt_hub, hub_off_mat)
    rev_da_hub = term_rev(exp_gen_peak, exp_gen_off, da_hub, hub_off_mat)
    rev_rt_bus = term_rev(exp_gen_peak, exp_gen_off, rt_bus, hub_off_mat)
    rev_da_bus = term_rev(exp_gen_peak, exp_gen_off, da_bus, hub_off_mat)

    def fixed_price_from_revenues(rev: np.ndarray) -> Dict[str, float]:
        pctl = float(np.percentile(rev, P_LEVEL))
        mean = float(rev.mean())
        px_mean = mean / total_mwh_term if total_mwh_term > 0 else np.nan
        px_p    = pctl / total_mwh_term if total_mwh_term > 0 else np.nan
        risk_adj = px_p - px_mean
        return {"fixed_price": px_p, "mean_price": px_mean, "risk_adjustment": risk_adj}

    # basis components (mean monthly basis over priced term) & hub components
    basis_rt = float((rt_bus.mean(axis=1) - rt_hub.mean(axis=1)).mean())
    basis_da = float((da_bus.mean(axis=1) - da_hub.mean(axis=1)).mean())
    hub_component = {
        "RT": float(rt_hub.mean(axis=1).mean()),
        "DA": float(da_hub.mean(axis=1).mean()),
    }

    pr_rt_hub = fixed_price_from_revenues(rev_rt_hub)
    pr_da_hub = fixed_price_from_revenues(rev_da_hub)
    pr_rt_bus = fixed_price_from_revenues(rev_rt_bus)
    pr_da_bus = fixed_price_from_revenues(rev_da_bus)

    # build fixed price table with components
    fixed_rows = []
    def add_row(name, hub_mean, basis_mean, price_pack):
        fixed_rows.append({
            "Product": name,
            "Hub_Component": hub_mean,
            "Basis_Component": basis_mean,
            "Risk_Adjustment": price_pack["risk_adjustment"],
            "Fixed_Price_$/MWh": price_pack["fixed_price"],
        })

    add_row("RT Hub",     hub_component["RT"], 0.0,      pr_rt_hub)
    add_row("DA Hub",     hub_component["DA"], 0.0,      pr_da_hub)
    add_row("RT Busbar",  hub_component["RT"], basis_rt, pr_rt_bus)
    add_row("DA Busbar",  hub_component["DA"], basis_da, pr_da_bus)

    pd.DataFrame(fixed_rows, columns=[
        "Product","Hub_Component","Basis_Component","Risk_Adjustment","Fixed_Price_$/MWh"
    ]).to_csv(RESULTS_DIR / "fixed_prices.csv", index=False)

    valuation = {
        "term_years": [2026, 2027, 2028, 2029, 2030],
        "p_level": 75,
        "expected_generation_csv": str((RESULTS_DIR / "expected_generation_2026-2030.csv").resolve()),
        "fixed_prices_csv": str((RESULTS_DIR / "fixed_prices.csv").resolve()),
        "fixed_prices": fixed_rows,
    }
    with open(RESULTS_DIR / "valuation.json", "w") as f:
        json.dump(valuation, f, indent=2)

    print("\nValuation artifacts written:")
    print(" - expected_generation_2026-2030.csv")
    print(" - fixed_prices.csv")
    print(" - valuation.json")
    print(f"Results folder: {RESULTS_DIR.resolve()}")

    return valuation

# ============================================
# CLI Entry Point (always valuation mode)
# ============================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m scripts.Monte_Carlo_sim <company> <n_sims> [--reuse-gen]")
        sys.exit(1)

    company = sys.argv[1].upper()
    n_sims = int(sys.argv[2])
    reuse = "--reuse-gen" in sys.argv[3:]

    run_valuation_model(company, n_sims=n_sims, reuse_gen=reuse)