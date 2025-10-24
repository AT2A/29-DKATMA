# scripts/plot_data.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CANON = {
    "gen": "Gen",
    "rt_busbar": "RT_Busbar",
    "rt_hub": "RT_Hub",
    "da_busbar": "DA_Busbar",
    "da_hub": "DA_Hub",
}

def _normalize_cols(cols):
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = "".join(ch if ch.isalnum() else "_" for ch in s)
        s = "_".join(filter(None, s.split("_")))
        out.append(s)
    return out

def _map_to_canon(df):
    # normalize names
    df = df.copy()
    df.columns = _normalize_cols(df.columns)
    # try to map variants to canonical keys
    mapping = {}
    for norm in df.columns:
        if norm in ("gen", "generation", "gen_mwh", "mwh"):
            mapping[norm] = "gen"
        elif "rt" in norm and "busbar" in norm:
            mapping[norm] = "rt_busbar"
        elif "rt" in norm and "hub" in norm:
            mapping[norm] = "rt_hub"
        elif ("da" in norm or "day_ahead" in norm) and "busbar" in norm:
            mapping[norm] = "da_busbar"
        elif ("da" in norm or "day_ahead" in norm) and "hub" in norm:
            mapping[norm] = "da_hub"
    # build a slim df with canonical column names (if present)
    slim = pd.DataFrame(index=df.index)
    for src, canon_key in mapping.items():
        slim[CANON[canon_key]] = pd.to_numeric(df[src], errors="coerce")
    return slim

def plot_file(csv_path: str | Path, resample="D", out_dir="plots", show=True):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    df = _map_to_canon(df)

    # keep only columns we have
    cols = [c for c in ["Gen","RT_Busbar","RT_Hub","DA_Busbar","DA_Hub"] if c in df.columns]
    if not cols:
        raise ValueError(f"No expected columns found in {csv_path.name}. Found: {list(df.columns)}")

    # resample for readability (D=daily, W=weekly, H=hourly/no-op)
    if resample:
        agg = {"Gen": "sum"}  # generation as energy → sum per day
        for c in cols:
            if c != "Gen":
                agg[c] = "mean"
        df = df[cols].resample(resample).agg(agg)

    # make stacked subplots
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    title = csv_path.stem.replace("_cleaned", "") + " Time Series"
    fig.suptitle(title)

    for ax, c in zip(axes, cols):
        ax.plot(df.index, df[c], linewidth=0.8)
        ax.set_ylabel(c)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Datetime")

    out_png = out_dir / f"{csv_path.stem}_subplots.png"
    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {out_png.resolve()}")

def plot_all(data_dir="data/cleaned", resample="D", show=True):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_cleaned.csv"))
    if not files:
        print("No cleaned CSVs found.")
        return
    for f in files:
        plot_file(f, resample=resample, show=show)

if __name__ == "__main__":
    # Daily resample is a sane default for multi-year hourly data.
    plot_all("data/cleaned", resample="D", show=True)
