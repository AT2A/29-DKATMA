# Avangrid Hackathon — Merchant Renewable Valuation Framework

> **Scope:** ERCOT, MISO (wind), CAISO (solar) — model generation, prices, and revenue risk for 2026–2030.  
> **Core outputs:** risk-adjusted fixed prices ($/MWh) at **P75**; monthly PEAK/OFFPEAK energy quantiles; simulated Busbar price paths; validation plots.

---

## 0) Repository Map

```
scripts/
  clean_data.py
  simulate_miso_gen.py
  CAISO_sim_framework.py
  MISO_sim_framework.py
  ERCOT_sim_framework.py
  Monte_Carlo_sim.py
  plot_miso_sim_compare.py
  plot_caiso_sim_compare.py

data/
  HackathonDataset.xlsx         # input
  cleaned/                      # outputs from clean_data.py
    ERCOT_cleaned.csv
    ERCOT_forwards.csv
    MISO_cleaned.csv
    MISO_forwards.csv
    CAISO_cleaned.csv
    CAISO_forwards.csv

projections_* / data/simulations/  # outputs from simulation runs
plots/                              # validation plots
```

---

## 1) Data Sources & Assumptions

- **Hourly historicals** (three sheets): `ERCOT`, `MISO`, `CAISO`  
  Fields include `Date`, `HE` (Hour Ending), `Gen` (MW), prices (`Hub`, `Busbar`, `DA/RT` variants).
- **Monthly forward curves** (per sheet): `Forward_Peak`, `Forward_OffPeak` for hub reference.
- **Ignore** Capex, O&M, REC/capacity value, taxes (per prompt). The exercise values **energy-only revenues**.
- **Risk appetite:** compute fixed price so that **fixed ≥ merchant** in **75%** of scenarios (P75).

---

## 2) Cleaning Pipeline (`scripts/clean_data.py`)

### 2.1 Header detection
- Automatically finds header row (first column labelled "Date") in each sheet.  
  This is robust to preambles and notes above the table.

### 2.2 Timestamp construction
- Combine **`Date` + `HE`** into a single **`Datetime`**:
  - `align="start"` → HE=1 maps to 00:00; HE=24 maps to 23:00 (start-of-hour convention).
  - Result is set as index and sorted.

### 2.3 Numeric coercion & missing values
- Coerce key numeric columns: `Gen`, `Forward_Peak`, `Forward_OffPeak`, and any extra requested.
- Fill policy:
  - `Gen`: NaN → 0 (conservative, avoids creating spurious generation).
  - **Price columns** (auto-detected by name: *hub*, *busbar*, *rt*, *da*, *price*): linear interpolate → `bfill`/`ffill`.
  - **Forwards**: same interpolate policy.
  - Optional: drop rows that **still** have *all* price columns missing after fills.

### 2.4 Outputs
- For each ISO:  
  - `*_cleaned.csv` (hourly indexed by `Datetime`)  
  - `*_forwards.csv` (monthly indexed by `Month` → first of month)

> These files are the canonical inputs to the generation and price simulations.

---

## 3) Generation Simulation

### 3.1 MISO & ERCOT (block bootstrap)

- **Method:** Month-aware, contiguous **block bootstrap** over historical hourly `Gen`:
  1. Split the history by **calendar month**.
  2. For each target month/year in 2026–2030:
     - Sample **contiguous blocks** of length `block_len` (typ. 6–12 h)
     - Use a **month window** `m ± month_band` (wraps Jan/Dec) to preserve seasonality.
     - Stitch blocks until the entire year is filled.
  3. **Capacity scaling** per year via:
     - `per_year_mult`: explicit factors (e.g., 2026:1.00 … 2030:1.08), **or**
     - `CAGR` relative to the first year, **or**
     - `base_mult` if neither is given.
- **Why it works:** Preserves intra-day autocorrelation and seasonal patterns with minimal assumptions.

**Scripts/Functions**
- `scripts/simulate_miso_gen.py` (CLI) — produces per-simulation hourly CSVs.
- `scripts/MISO_sim_framework.py::run_miso_gen_sim(...)`
- `scripts/ERCOT_sim_framework.py::run_ercot_gen_sim(...)`

**Key parameters**
- `block_len` — hours per sampled block.
- `month_band` — month window width (1 → {m-1, m, m+1}).
- `per_year_mult` or `cagr`/`base_mult` — annual scaling.

---

### 3.2 CAISO Solar (baseline × residuals + bootstrap)

- **Clear-sky baseline (envelope):**
  - For each `(day-of-year, hour)`, compute a **high quantile** (P98) of historical Gen → **upper envelope**.
  - Smooth across DOY with rolling window; fill to cover all **1..366** days.
- **Residuals:** `Residual = Actual / Baseline` (clipped to [0, 1.3]) captures cloud/curtailment variability.
- **Bootstrap:** Month-aware **block bootstrap** of **residuals**; multiply by **future baseline**.
- **Jitter:** Optional small Gaussian noise on high-baseline hours to avoid "too-perfect" curves.
- **Annual scaling:** Apply `per_year_mult` or `cagr` as with MISO/ERCOT.
- **Outputs:**
  - Hourly sims (optional, chunked)
  - **Monthly PEAK/OFFPEAK MWh quantiles** (P50/P75/P90) — fast downstream use in pricing/revenue.

**Scripts/Functions**
- `scripts/simulate_caiso_solar.py` (CLI) — end-to-end pipeline with saving & metadata JSON.
- `scripts/CAISO_sim_framework.py::run_caiso_gen_sim(...)` — same logic wrapped as a reusable function.

**Key parameters**
- `block_len`, `month_band` — residual block bootstrap knobs.
- `jitter_frac` — fraction of baseline for noise scale (typ. 0.01).
- `peak_mode` — `"caiso_like"` (Mon–Sat & 07:00–22:00) or `"simple"` (daily 07:00–22:00).

---

## 4) Price Simulation (Busbar)

**Goal:** Convert **Hub forward curves** into **Busbar price distributions** by adding **historical spread risk**.

- From cleaned hourly prices:
  - Compute **RT** and **DA** spreads (where present):  
    `Real_Spread = Busbar - Hub` (or `RT Busbar - RT Hub`)  
    `DA_Spread  = DA Busbar - DA Hub`  
  - Derive **Avg_Spread = (RT + DA)/2` as a balanced basis proxy.
  - Aggregate by **calendar month** → `(Spread_Mean, Spread_Std)`.
- Merge spreads with **monthly Hub forward** → simulate Busbar:
  \[
    \text{Busbar}_m = \text{Forward\_Peak}_m + \text{Spread\_Mean}_m + \mathcal{N}(0,\ \text{Spread\_Std}_m)
  \]

**Scripts/Functions**
- `scripts/MISO_sim_framework.py::run_miso_price_sim(...)`
- `scripts/ERCOT_sim_framework.py::run_ercot_price_sim(...)`
- `scripts/CAISO_sim_framework.py::run_caiso_price_sim(...)`

**Notes**
- If DA fields are missing for a market, fall back to **RT** spreads only.
- Per prompt, **use Peak forward** for fixed-price valuation; OffPeak is available for analysis.

---

## 5) Revenue Monte Carlo (`scripts/Monte_Carlo_sim.py`)

**Integration loop:** for `n_sims` simulations
1. **Price matrix:** Vectorized Busbar simulation per forward month → shape `[months, n_sims]`.
2. **Generation:** Either
   - `reuse_gen=True` → generate **once** and reuse across sims (faster), or
   - `reuse_gen=False` → **new** gen path per sim (full Monte Carlo).
3. **Monthly aggregation:** Sum hourly Gen to **monthly MWh**.
4. **Revenue per sim:** `Revenue_i = Σ (MWh_month × Price_month_i)`.
5. **Distribution summary:** mean, stdev, **P75** revenue.  
6. **Fixed price equivalents ($/MWh):**
   - `F_mean = mean(Revenue) / mean(total MWh)`
   - `F_P75  = P75(Revenue) / mean(total MWh)`

**Outputs (printed)**
- Revenue distribution stats (mean/σ/P75)
- Implied fixed prices at mean and P75
- Timing diagnostics (price prep / gen prep / per-sim / total)

---

## 6) Validation & Plots

### Historical vs Simulated — **shape sanity check**
- `scripts/plot_miso_sim_compare.py`
- `scripts/plot_caiso_sim_compare.py`

Panels:
1. Full historical hourly generation
2. Full simulated hourly generation (inferred horizon label)
3. **Overlap panel** (index-aligned): compares last *N* hours of historical with first *N* hours of simulated (optional rolling mean smoothing).

> Use this to tune `block_len`, `month_band`, and CAISO `jitter_frac` if the sim looks “too stiff” or “too noisy.”

---

## 7) CLI Usage — End‑to‑End

> All paths below assume you run from repo root.

### 7.1 Cleaning
```bash
python scripts/clean_data.py \
  -i data/HackathonDataset.xlsx \
  -o data/cleaned \
  --align start \
  --require-price
```

### 7.2 Generation (examples)
MISO (3 sims):
```bash
python scripts/simulate_miso_gen.py data/cleaned/MISO_cleaned.csv \
  --start_year 2026 --end_year 2030 \
  --block_len 8 --month_band 1 \
  --n_sims 3 --out_dir projections_miso \
  --per_year_mult "2026:1.00,2027:1.03,2028:1.06,2029:1.09,2030:1.12"
```
CAISO solar (1000 sims + monthly quantiles + hourly chunks):
```bash
python scripts/simulate_caiso_solar.py data/cleaned/CAISO_cleaned.csv \
  --start_year 2026 --end_year 2030 \
  --per_year_mult "2026:1.00,2027:1.02,2028:1.04,2029:1.06,2030:1.08" \
  --block_len 12 --month_band 1 --n_sims 1000 \
  --jitter_frac 0.01 --peak_mode caiso_like \
  --out_dir projections_caiso --save_hourly
```

### 7.3 Prices (single run preview)
```bash
python -c "from scripts.MISO_sim_framework import run_miso_price_sim; print(run_miso_price_sim(save=False).head())"
```

### 7.4 Revenue MC (risk‑adjusted fixed prices)
CAISO, 1000 sims, reuse the same gen path (fast exploratory run):
```bash
python scripts/Monte_Carlo_sim.py CAISO 1000 --reuse-gen --timings
```

---

## 8) Parameter Tuning & Guidance

- **`block_len`**: ↑ → smoother diurnal shape; ↓ → noisier, higher variance.
- **`month_band`**: ↑ → blends across more months; ↓ → tighter seasonality.
- **`per_year_mult` vs `cagr`**: prefer explicit `per_year_mult` per ISO if you have a programmatic ramp plan.
- **CAISO `jitter_frac`**: use small values (0.005–0.02) if simulated peaks look **too** deterministic vs history.
- **`reuse_gen`** in Monte Carlo: `True` for rapid pricing sweeps; `False` for full price × volume risk.

---

## 9) Peak / Off-Peak Definition

- **`caiso_like`** (default): PEAK = 07:00–22:00 **and** Mon–Sat. OFFPEAK = all else.  
- **`simple`**: PEAK = 07:00–22:00 **daily**.

The CAISO pipeline produces **monthly PEAK/OFFPEAK MWh quantiles** directly for valuation.

---

## 10) Troubleshooting Notes

- **“Simulation file must include a Datetime column.”**  
  When using `plot_caiso_sim_compare.py`, pass a **folder** containing `hourly_sims_*.csv` **or** a single hourly sim file; monthly summary files don’t have a `Datetime` column. Use the `--sim_col` flag if needed.

- **Duplicate timestamps in historicals (DST)**  
  The loader aggregates duplicates via mean/sum/first (configurable in MISO CLI). The cleaning step also builds a gap-free hourly index and interpolates small gaps safely.

- **ERCOT ‘generated 3 times’ confusion**  
  `--n_sims 3` in generation CLI **intentionally** creates **3 independent** Monte Carlo paths, saved as separate files.

- **Solar sim “not following trend”**  
  If the first few simulated weeks look off versus historical *last* weeks, that’s expected (the overlap panel aligns by **index**, not calendar). Tune `block_len`, `month_band`, and CAISO `jitter_frac`; ensure `per_year_mult` reflects intended ramp.

---

## 11) Limitations & Next Steps

- **Prices:** Only **Busbar** is simulated (Hub forward + historical spreads). Could extend to joint DA/RT modeling and negative price frequency explicitly.
- **Curtailment:** Implicit in history; no explicit curtailment model in future baseline beyond residual caps.
- **Correlation:** Gen and spreads are **independent** in MC. Future work: couple price spreads to gen/weather regimes.
- **Fixed price per period:** Framework computes **all-in** fixed price. Could split by PEAK/OFFPEAK tranches using CAISO monthly quantiles with separate forward curves.

**Enhancements**
- Add **basis regime** states (seasonal/structural shifts).
- Weather-conditioned residual sampling.
- Explicit **negative price** rules and imbalance settlement logic.
- Slide-ready tables and charts (auto-export for deck).

---

## 12) Reproducibility

- Scripts print config and can write a **settings JSON** (e.g., CAISO solar pipeline).
- Keep `requirements.txt` / `environment.yml` with:
  - `pandas`, `numpy`, `matplotlib`, `pyarrow` (optional), `openpyxl` (for Excel).

Example `requirements.txt`:
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
openpyxl>=3.1
pyarrow>=15.0
```

---

## 13) Deliverables Checklist (per prompt)

- [x] **Valuation Model** (2026–2030): expected gen (monthly PEAK/OFFPEAK); fixed prices ($/MWh) at Hub/Busbar, RT/DA variants supported by inputs.  
- [x] **Risk adjustments**: fully integrated via MC distributions.  
- [x] **Risk appetite P75**: computed.  
- [x] **Documentation**: this README plus script-level docstrings.  
- [x] **Reproducibility**: deterministic seeds, saved configs, organized outputs.  
- [ ] **Slide deck**: to be generated (plots & summary tables).

---

## 14) Quick Starts

**A) One-click CAISO end-to-end (1000 sims + monthly quantiles):**
```bash
python scripts/simulate_caiso_solar.py data/cleaned/CAISO_cleaned.csv \
  --start_year 2026 --end_year 2030 \
  --per_year_mult "2026:1.00,2027:1.02,2028:1.04,2029:1.06,2030:1.08" \
  --block_len 12 --month_band 1 --n_sims 1000 \
  --jitter_frac 0.01 --peak_mode caiso_like \
  --out_dir projections_caiso --save_hourly
python scripts/Monte_Carlo_sim.py CAISO 1000 --reuse-gen --timings
```

**B) MISO 3-path sanity check + plot:**
```bash
python scripts/simulate_miso_gen.py data/cleaned/MISO_cleaned.csv \
  --start_year 2026 --end_year 2030 \
  --block_len 8 --month_band 1 --n_sims 3 \
  --out_dir projections_miso \
  --per_year_mult "2026:1.00,2027:1.03,2028:1.06,2029:1.09,2030:1.12"

python scripts/plot_miso_sim_compare.py \
  data/cleaned/MISO_cleaned.csv \
  projections_miso/MISO_sim_hourly_2026_2030_sim1.csv \
  --overlap_len_hours $((24*30)) --overlap_smooth_h 6 \
  --out_png plots/miso_hist_vs_sim.png
```

**C) CAISO plot from hourly chunks folder:**
```bash
python scripts/plot_caiso_sim_compare.py \
  data/cleaned/CAISO_cleaned.csv \
  projections_caiso \
  --sim_col sim_0 \
  --out_png plots/caiso_hist_vs_sim.png
```

---

**Author’s note:** This README consolidates and explains the current codebase behavior and intended usage. It is designed to be pasted directly into the repository root.
