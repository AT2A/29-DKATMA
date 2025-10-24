# scripts/load_data.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Helpers
# -----------------------------
def detect_header_row(raw_df: pd.DataFrame) -> int:
    """Find the row index containing 'Date' in the first column."""
    col0 = raw_df.iloc[:, 0].astype(str).str.strip().str.lower()
    matches = raw_df.index[col0 == "date"]
    if len(matches) == 0:
        raise ValueError("Could not find header row containing 'Date' in first column.")
    return int(matches[0])

def combine_date_he(df: pd.DataFrame, date_col="Date", he_col="HE", align="start") -> pd.DataFrame:
    """
    Combine Date + HE (Hour Ending) into a single Datetime column.
    align='start' -> timestamp at the start of the hour (HE=1 -> 00:00)
    align='end'   -> timestamp at the end of the hour  (HE=1 -> 01:00)
    """
    if date_col not in df.columns or he_col not in df.columns:
        missing = [c for c in (date_col, he_col) if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[he_col] = pd.to_numeric(df[he_col], errors="coerce", downcast="integer")

    # Drop rows that cannot form a timestamp
    df = df.dropna(subset=[date_col, he_col]).copy()

    if align == "start":
        df["Datetime"] = df[date_col] + pd.to_timedelta(df[he_col] - 1, unit="h")
    elif align == "end":
        df["Datetime"] = df[date_col] + pd.to_timedelta(df[he_col], unit="h")
    else:
        raise ValueError("align must be 'start' or 'end'")

    df = df.set_index("Datetime").sort_index()
    return df

def coerce_numeric_columns(df: pd.DataFrame, numeric_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Convert listed columns to numeric if present."""
    if numeric_cols is None:
        numeric_cols = ["Gen", "Forward_Peak", "Forward_OffPeak"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def identify_price_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristic: price columns usually include 'Hub', 'Busbar', 'RT', 'DA', or 'Price'
    but exclude forward curve columns explicitly handled elsewhere.
    """
    candidates = []
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ["hub", "busbar", "rt", "da", "price"]):
            if not name.startswith("forward"):  # keep forwards separate
                # exclude non-numeric columns that happen to match
                if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == "object":
                    candidates.append(c)
    return candidates

def fill_missing_values(
    df: pd.DataFrame,
    fill_gen_zero: bool = True,
    interpolate_prices: bool = True,
    interpolate_forwards: bool = True,
    forward_cols: Iterable[str] = ("Forward_Peak", "Forward_OffPeak"),
    require_price_any: bool = False,
) -> pd.DataFrame:
    """
    Fill missing values using the agreed policy:
    - Gen -> fill NaN with 0
    - Price columns -> linear interpolate, then bfill/ffill
    - Forward curve columns -> linear interpolate, then bfill/ffill
    - Optionally drop rows where ALL price columns remain NaN after fills
    """
    df = df.copy()

    # Gen
    if fill_gen_zero and "Gen" in df.columns:
        df["Gen"] = pd.to_numeric(df["Gen"], errors="coerce").fillna(0)

    # Prices
    price_cols = identify_price_columns(df)  # dynamic detection
    if interpolate_prices and price_cols:
        df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
        df[price_cols] = df[price_cols].interpolate(method="linear", limit_direction="both")
        df[price_cols] = df[price_cols].bfill().ffill()

    # Forwards
    fwd_cols_present = [c for c in forward_cols if c in df.columns]
    if interpolate_forwards and fwd_cols_present:
        df[fwd_cols_present] = df[fwd_cols_present].apply(pd.to_numeric, errors="coerce")
        df[fwd_cols_present] = df[fwd_cols_present].interpolate(method="linear", limit_direction="both")
        df[fwd_cols_present] = df[fwd_cols_present].bfill().ffill()

    # Optional: drop rows if ALL price columns are still NaN (should be rare)
    if require_price_any and price_cols:
        mask_all_price_nan = df[price_cols].isna().all(axis=1)
        dropped = int(mask_all_price_nan.sum())
        if dropped:
            logger.warning(f"Dropping {dropped} rows with all price columns NaN after fills.")
        df = df[~mask_all_price_nan]

    return df

# -----------------------------
# Main sheet loader
# -----------------------------
def load_and_clean_sheet(
    file_path: Path,
    sheet_name: str,
    numeric_cols: Optional[Iterable[str]] = None,
    align: str = "start",
    require_price_any: bool = False,
) -> pd.DataFrame:
    """
    Load and clean one sheet:
      - auto-detect header row
      - drop empty columns
      - coerce known numeric cols
      - drop rows missing Date/HE (only)
      - combine Date+HE into Datetime index
      - fill Gen=0, interpolate price & forward columns
    """
    logger.info(f"Loading sheet: {sheet_name}")

    # Detect header
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_row = detect_header_row(raw_df)

    # Reload with header and drop fully empty columns
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
    df = df.dropna(axis=1, how="all")

    # Coerce numerics user requested (Gen/Forward_*)
    df = coerce_numeric_columns(df, numeric_cols)

    # Combine timestamp (this also drops only rows missing Date/HE)
    df = combine_date_he(df, "Date", "HE", align)

    # Fill NaNs according to policy
    df = fill_missing_values(
        df,
        fill_gen_zero=True,
        interpolate_prices=True,
        interpolate_forwards=True,
        forward_cols=("Forward_Peak", "Forward_OffPeak"),
        require_price_any=require_price_any,
    )

    logger.info(f"{sheet_name}: cleaned {len(df)} rows; columns: {list(df.columns)}")
    return df

def load_monthly_forwards(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Extract monthly forward prices (projected Peak/OffPeak) from columns K–M (rows 11–70).
    Expected columns:
        K -> Month (e.g., 'Jan-26')
        L -> Forward_Peak
        M -> Forward_OffPeak
    """
    logger.info(f"Loading monthly forwards from {sheet_name}...")

    # Read just the range (usecols limits columns, skip rows before 10)
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        usecols="K:M",
        skiprows=10,  # zero-based index, skips top 10 rows (so row 11 = index 10)
        nrows=60,     # expected number of months
        names=["Month", "Forward_Peak", "Forward_OffPeak"],
    )

    # Drop rows without a valid month label
    df = df.dropna(subset=["Month"])

    # Convert Month like "Jan-26" -> datetime (first day of month)
    df["Month"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")

    # Drop rows that couldn’t parse
    df = df.dropna(subset=["Month"])

    # Coerce prices to numeric
    df["Forward_Peak"] = pd.to_numeric(df["Forward_Peak"], errors="coerce")
    df["Forward_OffPeak"] = pd.to_numeric(df["Forward_OffPeak"], errors="coerce")

    # Set Datetime index
    df = df.set_index("Month").sort_index()

    logger.info(f"{sheet_name} monthly forwards: {len(df)} months loaded.")
    return df

def load_and_clean_data(
    file_path: str | Path,
    sheet_names: Iterable[str] = ("ERCOT", "MISO", "CAISO"),
    numeric_cols: Optional[Iterable[str]] = None,
    align: str = "start",
    require_price_any: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load and clean both hourly and monthly forward data for each sheet.
    Returns nested dict:
      data[ISO]["hourly"]
      data[ISO]["forwards"]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    cleaned: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sheet in sheet_names:
        hourly_df = load_and_clean_sheet(
            path, sheet,
            numeric_cols=numeric_cols,
            align=align,
            require_price_any=require_price_any,
        )
        monthly_df = load_monthly_forwards(path, sheet)
        cleaned[sheet] = {
            "hourly": hourly_df,
            "forwards": monthly_df,
        }
    return cleaned

def save_cleaned_data(data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str | Path = "data/cleaned") -> None:
    """Save hourly and monthly cleaned data for each ISO."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, parts in data.items():
        hourly_path = out / f"{name}_cleaned.csv"
        parts["hourly"].to_csv(hourly_path)
        logger.info(f"Saved hourly data: {hourly_path.resolve()}")

        monthly_path = out / f"{name}_forwards.csv"
        parts["forwards"].to_csv(monthly_path)
        logger.info(f"Saved monthly forwards: {monthly_path.resolve()}")

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Clean HackathonDataset.xlsx, combine Date+HE, and fill NaNs.")
    parser.add_argument("-i", "--input", default="data/HackathonDataset.xlsx", help="Path to Excel file.")
    parser.add_argument("-s", "--sheets", nargs="+", default=["ERCOT", "MISO", "CAISO"], help="Sheets to load.")
    parser.add_argument("-o", "--output", default="data/cleaned", help="Output directory for cleaned CSVs.")
    parser.add_argument("--align", choices=["start", "end"], default="start", help="Align HE as start or end of hour.")
    parser.add_argument("--extra-cols", nargs="*", default=[], help="Extra numeric columns to coerce.")
    parser.add_argument("--require-price", action="store_true",
                        help="Drop rows that still have all price columns NaN after interpolation/fill.")
    args = parser.parse_args()

    numeric_cols = ["Gen", "Forward_Peak", "Forward_OffPeak"] + args.extra_cols
    cleaned = load_and_clean_data(
        args.input,
        sheet_names=args.sheets,
        numeric_cols=numeric_cols,
        align=args.align,
        require_price_any=args.require_price,
    )
    save_cleaned_data(cleaned, args.output)

if __name__ == "__main__":
    main()
