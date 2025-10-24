from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================
# MISO PRICE SIMULATION (single run)
# ============================================
def run_miso_price_sim(
    cleaned_dir: str | Path = "data/cleaned",
    seed: int | None = None,
    save: bool = False,
    output_dir: str | Path = "data/simulations"
) -> pd.DataFrame:
    """Run a single Monte Carlo simulation projecting MISO Busbar prices."""
    rng = np.random.default_rng(seed)

    cleaned_dir = Path(cleaned_dir)
    cleaned_price_path = cleaned_dir / "MISO_cleaned.csv"
    forwards_path      = cleaned_dir / "MISO_forwards.csv"
    if not cleaned_price_path.exists():
        raise FileNotFoundError(f"Missing file: {cleaned_price_path}")
    if not forwards_path.exists():
        raise FileNotFoundError(f"Missing file: {forwards_path}")

    print(f"[MISO_price] reading: {cleaned_price_path}")
    print(f"[MISO_price] reading: {forwards_path}")

    df  = pd.read_csv(cleaned_price_path, parse_dates=["Datetime"])
    fwd = pd.read_csv(forwards_path,     parse_dates=["Month"])

    df["Real_Spread"] = df["Busbar"] - df["Hub"]
    df["DA_Spread"]   = df["DA Busbar"] - df["DA Hub"]
    df["Avg_Spread"]  = (df["Real_Spread"] + df["DA_Spread"]) / 2
    df["Month"]       = df["Datetime"].dt.month

    monthly_spread_stats = (
        df.groupby("Month")["Avg_Spread"]
          .agg(["mean", "std"])
          .rename(columns={"mean": "Spread_Mean", "std": "Spread_Std"})
          .reset_index()
    )

    fwd["Month"] = fwd["Month"].dt.month
    proj = fwd.merge(monthly_spread_stats, on="Month", how="left")
    proj["Noise"] = rng.normal(0, proj["Spread_Std"])
    proj["Busbar_Projected"] = proj["Forward_Peak"] + proj["Spread_Mean"] + proj["Noise"]

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "MISO_Busbar_Simulation_single.csv"
        proj.to_csv(out_path, index=False)
        print(f"✅ Saved single Busbar simulation to {out_path.resolve()}")

    return proj


# ============================================
# MISO GENERATION SIMULATION (optimized)
# ============================================
def run_miso_gen_sim(
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
    """Run a synthetic MISO generation simulation using block bootstrapping."""
    rng = np.random.default_rng(seed)

    # ---- read once or reuse ----
    if hist_series is None:
        if input_csv is None:
            raise ValueError("Must provide either hist_series or input_csv.")
        input_csv = Path(input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"Missing file: {input_csv}")
        print(f"[MISO_gen ] reading: {input_csv}")
        df = pd.read_csv(input_csv, parse_dates=["Datetime"])
        gen = pd.to_numeric(df["Gen"], errors="coerce")
        hist_series = pd.Series(gen.values, index=df["Datetime"]).dropna()

    hist = hist_series
    hist_by_month = {m: hist[hist.index.month == m] for m in range(1, 13)}

    years = list(range(start_year, end_year + 1))
    y0 = years[0]
    year_mults = {y: base_mult * ((1 + (cagr or 0.0)) ** (y - y0)) for y in years}

    all_parts = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        end   = pd.Timestamp(f"{y}-12-31 23:00")
        idx   = pd.date_range(start, end, freq="h")

        vals, pos = [], 0
        while pos < len(idx):
            m = idx[pos].month
            months_pool = {((m - 1 + d) % 12) + 1 for d in range(-month_band, month_band + 1)}
            candidates = [t for mm in months_pool for t in hist_by_month.get(mm, pd.Series(dtype=float)).index]
            if not candidates:
                candidates = list(hist.index)
            start_ts = pd.Timestamp(rng.choice(candidates)).floor("h")
            try:
                loc = hist.index.get_loc(start_ts)
                if isinstance(loc, slice):
                    loc = hist.index.slice_indexer(start_ts, start_ts).start
            except KeyError:
                continue
            block = hist.iloc[loc:loc + block_len].values
            take  = min(block_len, len(idx) - pos)
            vals.extend(block[:take])
            pos += take

        vals_arr = np.asarray(vals, dtype=float)
        if len(vals_arr) < len(idx):
            vals_arr = np.concatenate([vals_arr, np.full(len(idx) - len(vals_arr), vals_arr[-1])])
        elif len(vals_arr) > len(idx):
            vals_arr = vals_arr[:len(idx)]

        sim_year = pd.Series(vals_arr, index=idx) * year_mults[y]
        all_parts.append(sim_year)

    sim = pd.concat(all_parts).rename("Gen")

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"MISO_Gen_Simulation_{start_year}_{end_year}.csv"
        sim.to_csv(out_path, index_label="Datetime")
        print(f"Saved single Gen simulation to {out_path.resolve()}")

    return sim


if __name__ == "__main__":
    price_df = run_miso_price_sim(save=False)
    gen_df   = run_miso_gen_sim(save=False)
    print("\nFinished one MISO price & generation simulation run.")