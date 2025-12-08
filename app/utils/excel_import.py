# excel_import_utils.py
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


# ─────────────────────────────
# 1) Column name normalization
# ─────────────────────────────
def normalize_column_name(name: str) -> str:
    """
    Turn arbitrary Excel header into a safe SQL identifier:
    - lowercased
    - spaces/punctuation -> underscores
    - no leading digit
    """
    if not name:
        name = "column"

    name = name.strip().lower()
    # replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name)
    # remove leading/trailing underscores
    name = name.strip("_")
    if not name:
        name = "column"

    # avoid leading digit
    if re.match(r"^\d", name):
        name = f"c_{name}"

    return name


def make_unique_names(names: List[str]) -> List[str]:
    seen = {}
    result = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 0
            result.append(base)
        else:
            seen[base] += 1
            new_name = f"{base}_{seen[base]}"
            result.append(new_name)
    return result


# ─────────────────────────────
# 2) Type inference
# ─────────────────────────────
def try_parse_int(x: Any) -> Optional[int]:
    try:
        return int(str(x))
    except Exception:
        return None


def try_parse_float(x: Any) -> Optional[float]:
    try:
        return float(str(x))
    except Exception:
        return None


def try_parse_date(x: Any) -> Optional[datetime]:
    """
    Try a couple of common date formats. You can expand this list for your data.
    """
    if pd.isna(x):
        return None

    text = str(x).strip()
    if not text:
        return None

    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def infer_column_type(series: pd.Series, sample_size: int = 200) -> Dict[str, Any]:
    """
    Infer SQLite and logical type for a column based on a sample of values.
    Returns dict: {"sqlite_type": ..., "logical_type": ..., "date_format": ...?}
    """
    non_null = series.dropna()
    if non_null.empty:
        return {"sqlite_type": "TEXT", "logical_type": "string"}

    sample = non_null.sample(min(sample_size, len(non_null)), random_state=0)

    int_count = 0
    float_count = 0
    date_count = 0
    boolish_count = 0
    total = len(sample)

    for v in sample:
        s = str(v).strip().lower()

        if s in ("true", "false", "yes", "no", "y", "n", "0", "1"):
            boolish_count += 1

        if try_parse_int(v) is not None:
            int_count += 1
        elif try_parse_float(v) is not None:
            float_count += 1

        if try_parse_date(v) is not None:
            date_count += 1

    # Heuristics (tweak as needed)
    if date_count / total > 0.8:
        return {
            "sqlite_type": "TEXT",  # SQLite stores dates as TEXT
            "logical_type": "date",
            "date_format": "YYYY-MM-DD",  # we will normalize later
        }

    if boolish_count / total > 0.8:
        return {"sqlite_type": "INTEGER", "logical_type": "boolean"}

    if (int_count + float_count) / total > 0.8:
        if float_count > 0:
            return {"sqlite_type": "REAL", "logical_type": "number"}
        else:
            return {"sqlite_type": "INTEGER", "logical_type": "number"}

    return {"sqlite_type": "TEXT", "logical_type": "string"}


# ─────────────────────────────
# 3) Value cleaning
# ─────────────────────────────
EXCEL_ERROR_VALUES = {"#N/A", "#DIV/0!", "#REF!", "#NAME?", "#NULL!", "#VALUE!"}


def clean_value(value: Any, logical_type: str, meta: Dict[str, Any]) -> Any:
    """
    Normalize cell values before inserting into SQLite, based on inferred type.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    if s in EXCEL_ERROR_VALUES:
        return None

    if logical_type == "number":
        # meta doesn't distinguish int/float, SQLite handles REAL/INTEGER OK
        i = try_parse_int(s)
        if i is not None:
            return i
        f = try_parse_float(s)
        if f is not None:
            return f
        return None  # fallback

    if logical_type == "date":
        dt = try_parse_date(s)
        if dt is not None:
            # store ISO date string
            return dt.strftime("%Y-%m-%d")
        return None

    if logical_type == "boolean":
        if s in ("true", "yes", "y", "1"):
            return 1
        if s in ("false", "no", "n", "0"):
            return 0
        return None

    # default: string
    return s
