from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# ================================================================
# Visualization Helpers
# ================================================================

def plot_monthly_gen_band(monthly_gen_mat: pd.DataFrame, save_path: Path | None = None):
    # monthly_gen_mat: index = MonthStart, cols = Sim1..SimN (values are MWh)
    mean = monthly_gen_mat.mean(axis=1)
    std = monthly_gen_mat.std(axis=1)
    plt.figure(figsize=(9,4))
    plt.plot(monthly_gen_mat.index, mean, lw=1.5, label='Mean Monthly Gen (MWh)')
    plt.fill_between(monthly_gen_mat.index, mean - std, mean + std, alpha=0.2, label='±1 Std Dev')
    plt.title('Simulated Monthly Generation by Month')
    plt.xlabel('MonthStart'); plt.ylabel('MWh'); plt.legend(); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path / 'monthly_gen_band.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_price_hist(busbar_mat: pd.DataFrame, save_path: Path | None = None):
    # Flatten all prices across months & sims
    vals = busbar_mat.to_numpy().ravel()
    plt.figure(figsize=(7,4))
    plt.hist(vals, bins=50, alpha=0.75, edgecolor='black')
    plt.title('Histogram of Simulated Busbar Prices (All Months & Sims)')
    plt.xlabel('$/MWh'); plt.ylabel('Frequency'); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path / 'price_histogram.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_forward_vs_sim(busbar_mat: pd.DataFrame, save_path: Path | None = None):
    plt.figure(figsize=(9,4))
    plt.plot(busbar_mat.index, busbar_mat.mean(axis=1), lw=1.5, label='Mean Simulated Busbar')
    plt.fill_between(busbar_mat.index,
                     busbar_mat.mean(axis=1) - busbar_mat.std(axis=1),
                     busbar_mat.mean(axis=1) + busbar_mat.std(axis=1),
                     alpha=0.2, label='±1 Std Dev')
    plt.title('Simulated Busbar Prices by Month')
    plt.xlabel('Month'); plt.ylabel('$/MWh'); plt.legend(); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path / 'forward_vs_sim.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_revenue_distribution(revenues: np.ndarray, save_path: Path | None = None):
    plt.figure(figsize=(7,4))
    plt.hist(revenues, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.percentile(revenues,75), color='r', linestyle='--', label='75th pct')
    plt.legend(); plt.title('Revenue Distribution')
    plt.xlabel('Total Revenue ($)'); plt.ylabel('Frequency'); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path / 'revenue_distribution.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_fixed_price_distribution(revenues: np.ndarray, total_gen_each_sim: np.ndarray, save_path: Path | None = None):
    fixed_prices = revenues / total_gen_each_sim
    plt.figure(figsize=(7,4))
    plt.hist(fixed_prices, bins=40, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(np.percentile(fixed_prices,75), color='r', linestyle='--', label='75th pct')
    plt.legend(); plt.title('Simulated Fixed Price Distribution')
    plt.xlabel('$/MWh'); plt.ylabel('Frequency'); plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path / 'fixed_price_distribution.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_convergence(revenues: np.ndarray, save_path: Path | None = None):
    running = [np.percentile(revenues[:i+1],75) for i in range(len(revenues))]
    plt.figure(figsize=(7,4))
    plt.plot(running, lw=1.5)
    plt.title('Convergence of 75th Percentile Revenue Estimate')
    plt.xlabel('Simulations'); plt.ylabel('P75 Revenue ($)')
    plt.grid(alpha=0.3)
    if save_path: plt.savefig(save_path / 'convergence.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

# ================================================================
# Main Verification Routine
# ================================================================

def load_latest_results(company: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    results_root = root / 'data' / 'results' / company.upper()
    if not results_root.exists():
        raise FileNotFoundError(f'No results found for {company} in {results_root}')
    subfolders = [p for p in results_root.iterdir() if p.is_dir()]
    if not subfolders:
        raise FileNotFoundError(f'No result folders found in {results_root}')
    latest = max(subfolders, key=lambda p: p.stat().st_mtime)
    return latest

def verify_iso(company: str = 'MISO', results_dir: str | None = None):
    company = company.upper()
    if results_dir is None or results_dir == 'latest':
        folder = load_latest_results(company)
    else:
        folder = Path(results_dir)
    if not folder.exists():
        raise FileNotFoundError(f'Results folder not found: {folder}')

    print(f'\n===== Verifying {company} results from {folder.name} =====')

    # Load data
    revenues = np.loadtxt(folder / 'revenues.csv', delimiter=',')
    total_gen = np.loadtxt(folder / 'generation_summary.csv', delimiter=',')
    busbar_mat = pd.read_csv(folder / 'busbar_matrix.csv', index_col='MonthStart', parse_dates=['MonthStart'])
    # Optional: monthly generation matrix
    monthly_gen_mat = None
    mg_path = folder / 'monthly_gen_matrix.csv'
    if mg_path.exists():
        monthly_gen_mat = pd.read_csv(mg_path, index_col='MonthStart', parse_dates=['MonthStart'])
    summary = json.load(open(folder / 'summary.json'))

    # Print summary stats
    print('\n--- Simulation Summary ---')
    for k,v in summary.items():
        print(f'{k:20s}: {v}')

    # Generate plots
    plot_forward_vs_sim(busbar_mat, folder)
    plot_price_hist(busbar_mat, folder)
    plot_revenue_distribution(revenues, folder)
    plot_fixed_price_distribution(revenues, total_gen, folder)
    if monthly_gen_mat is not None:
        plot_monthly_gen_band(monthly_gen_mat, folder)
    plot_convergence(revenues, folder)

    print(f"\n✅ Verification complete. Plots saved to {folder.resolve()}")

if __name__ == '__main__':
    company = sys.argv[1] if len(sys.argv) > 1 else 'MISO'
    results_dir = sys.argv[2] if len(sys.argv) > 2 else 'latest'
    verify_iso(company, results_dir)