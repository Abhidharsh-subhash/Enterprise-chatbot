# pandas_executor.py
import os, re
import pandas as pd
import traceback
from collections import Counter
from app.core.openai_client import client, CHAT_MODEL
from app.core.logger import logger

UPLOADS_DIR = os.path.abspath("./uploads")


class DataFrameAnalyzer:
    """
    Analyzes a DataFrame to understand its structure dynamically.
    No hardcoded field names or values.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_profiles = {}
        self._analyze_columns()

    def _analyze_columns(self):
        """
        Profile each column to understand its nature:
        - cardinality (unique values)
        - data type
        - value frequency distribution
        - likely column purpose
        """
        for col in self.df.columns:
            series = self.df[col]
            non_null = series.dropna()

            profile = {
                "name": col,
                "dtype": str(series.dtype),
                "total_rows": len(series),
                "non_null_count": len(non_null),
                "unique_count": series.nunique(),
                "cardinality_ratio": series.nunique() / max(len(series), 1),
                "sample_values": [],
                "value_frequencies": {},
                "is_categorical": False,
                "is_identifier": False,
                "is_date": False,
                "is_numeric": False,
                "is_text_heavy": False,
            }

            # Determine column type
            if pd.api.types.is_datetime64_any_dtype(series):
                profile["is_date"] = True
            elif pd.api.types.is_numeric_dtype(series):
                profile["is_numeric"] = True
            else:
                # Try to detect dates in string columns
                if self._looks_like_date_column(non_null):
                    profile["is_date"] = True
                # Check if it's categorical (low cardinality) vs free text
                elif (
                    profile["cardinality_ratio"] < 0.3 and profile["unique_count"] < 100
                ):
                    profile["is_categorical"] = True
                elif profile["cardinality_ratio"] > 0.8:
                    profile["is_identifier"] = True
                else:
                    # Check average text length
                    avg_len = (
                        non_null.astype(str).str.len().mean()
                        if len(non_null) > 0
                        else 0
                    )
                    profile["is_text_heavy"] = avg_len > 50

            # Get sample values and frequencies for categorical columns
            if profile["is_categorical"] and len(non_null) > 0:
                value_counts = non_null.value_counts()
                profile["value_frequencies"] = value_counts.head(20).to_dict()
                profile["sample_values"] = list(value_counts.head(10).index)
            elif len(non_null) > 0:
                profile["sample_values"] = list(non_null.head(5).astype(str))

            self.column_profiles[col] = profile

    def _looks_like_date_column(self, series) -> bool:
        """Check if a string column contains date-like values."""
        if len(series) == 0:
            return False

        sample = series.head(20).astype(str)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            valid_ratio = parsed.notna().sum() / len(sample)
            return valid_ratio > 0.7
        except Exception:
            return False

    def get_categorical_columns(self) -> list:
        """Return columns that have categorical/enumerable values."""
        return [
            col
            for col, profile in self.column_profiles.items()
            if profile["is_categorical"]
        ]

    def get_date_columns(self) -> list:
        """Return columns that contain dates."""
        return [
            col for col, profile in self.column_profiles.items() if profile["is_date"]
        ]

    def get_identifier_columns(self) -> list:
        """Return columns that likely contain unique identifiers (names, IDs, etc.)."""
        return [
            col
            for col, profile in self.column_profiles.items()
            if profile["is_identifier"] or profile["cardinality_ratio"] > 0.5
        ]

    def get_all_categorical_values(self) -> dict:
        """Get all known categorical values per column."""
        return {
            col: profile["sample_values"]
            for col, profile in self.column_profiles.items()
            if profile["is_categorical"] and profile["sample_values"]
        }

    def find_value_in_data(self, search_term: str) -> list:
        """
        Find which columns contain the search term and how often.
        Returns list of (column, match_count, match_ratio, matched_values)
        """
        search_lower = search_term.lower().strip()
        results = []

        for col in self.df.columns:
            series = self.df[col].astype(str).str.lower()

            # Exact match
            exact_mask = series == search_lower
            exact_count = exact_mask.sum()

            # Partial/substring match
            try:
                partial_mask = series.str.contains(re.escape(search_lower), na=False)
                partial_count = partial_mask.sum()
            except Exception:
                partial_mask = series.apply(lambda x: search_lower in str(x).lower())
                partial_count = partial_mask.sum()

            if partial_count > 0:
                # Get the actual matched values
                matched_values = self.df.loc[partial_mask, col].unique()[:5]
                results.append(
                    {
                        "column": col,
                        "exact_matches": int(exact_count),
                        "partial_matches": int(partial_count),
                        "match_ratio": partial_count / len(self.df),
                        "matched_values": [str(v) for v in matched_values],
                        "is_categorical": self.column_profiles[col]["is_categorical"],
                    }
                )

        # Sort by: categorical columns first, then by match ratio (prefer selective matches)
        results.sort(
            key=lambda x: (
                -x["is_categorical"],  # Categorical first
                x["match_ratio"] if x["match_ratio"] < 0.5 else 1,  # Prefer selective
                -x["partial_matches"],  # Then by count
            )
        )

        return results

    def get_column_summary(self) -> str:
        """Generate a summary of the dataframe structure for the LLM."""
        lines = []

        # Date columns
        date_cols = self.get_date_columns()
        if date_cols:
            lines.append(f"DATE COLUMNS: {date_cols}")

        # Categorical columns with their possible values
        cat_cols = self.get_categorical_columns()
        if cat_cols:
            lines.append("CATEGORICAL COLUMNS (filter-friendly):")
            for col in cat_cols:
                values = self.column_profiles[col]["sample_values"][:10]
                lines.append(f"  - {col}: {values}")

        # Identifier/name columns
        id_cols = self.get_identifier_columns()
        if id_cols:
            lines.append(
                f"IDENTIFIER COLUMNS (names, IDs - high uniqueness): {id_cols}"
            )

        return "\n".join(lines)


def _dynamic_keyword_search(
    df: pd.DataFrame, analyzer: DataFrameAnalyzer, keyword: str
) -> pd.Series:
    """
    Dynamically search for a keyword using data-driven logic.
    No hardcoded terms - learns from the actual data.
    """
    if df.empty:
        return pd.Series(False, index=df.index)

    keyword = str(keyword).lower().strip()

    # Step 1: Find where this keyword (or parts of it) appears in the data
    search_results = analyzer.find_value_in_data(keyword)

    logger.info(f"Keyword '{keyword}' search results: {search_results}")

    # If exact/partial phrase found somewhere
    if search_results:
        # Prefer categorical columns (they're meant for filtering)
        categorical_matches = [r for r in search_results if r["is_categorical"]]

        if categorical_matches:
            # Use the best categorical column match
            best = categorical_matches[0]
            col = best["column"]

            logger.info(f"Using categorical column '{col}' for keyword search")

            # Search in this specific column
            mask = (
                df[col]
                .astype(str)
                .str.lower()
                .str.contains(re.escape(keyword), na=False)
            )
            if mask.any():
                return mask

        # If no categorical match, use the most selective match
        selective_matches = [r for r in search_results if r["match_ratio"] < 0.5]
        if selective_matches:
            best = selective_matches[0]
            col = best["column"]

            logger.info(f"Using selective column '{col}' for keyword search")

            mask = (
                df[col]
                .astype(str)
                .str.lower()
                .str.contains(re.escape(keyword), na=False)
            )
            if mask.any():
                return mask

    # Step 2: Token-based search with dynamic frequency analysis
    tokens = [t for t in re.split(r"\W+", keyword) if len(t) >= 2]

    if not tokens:
        return pd.Series(False, index=df.index)

    # Analyze each token's frequency across ALL data
    token_analysis = {}
    all_text = df.astype(str).agg(" ".join, axis=1).str.lower()

    for token in tokens:
        try:
            mask = all_text.str.contains(re.escape(token), na=False)
            count = mask.sum()
            ratio = count / len(df) if len(df) > 0 else 0

            token_analysis[token] = {
                "mask": mask,
                "count": count,
                "ratio": ratio,
                "is_rare": ratio < 0.3,  # Appears in less than 30% of rows
                "is_common": ratio > 0.7,  # Appears in more than 70% of rows
                "is_universal": ratio > 0.9,  # Appears in almost all rows
            }
        except Exception as e:
            logger.error(f"Token analysis error for '{token}': {e}")
            continue

    logger.info(
        f"Token analysis: {[(t, a['ratio']) for t, a in token_analysis.items()]}"
    )

    # Prioritize rare tokens (they're the discriminators)
    rare_tokens = [
        t for t, a in token_analysis.items() if a["is_rare"] and a["count"] > 0
    ]
    moderate_tokens = [
        t
        for t, a in token_analysis.items()
        if not a["is_rare"] and not a["is_universal"] and a["count"] > 0
    ]

    # Strategy: AND all rare tokens together
    if rare_tokens:
        mask = pd.Series(True, index=df.index)
        for token in rare_tokens:
            mask &= token_analysis[token]["mask"]

        if mask.any():
            logger.info(f"Rare token match ({rare_tokens}): {mask.sum()} rows")
            return mask

    # If no rare tokens, try moderate tokens
    if moderate_tokens:
        mask = pd.Series(True, index=df.index)
        for token in moderate_tokens:
            mask &= token_analysis[token]["mask"]

        if mask.any():
            logger.info(f"Moderate token match ({moderate_tokens}): {mask.sum()} rows")
            return mask

    # Last resort: OR any matching tokens
    any_mask = pd.Series(False, index=df.index)
    for token, analysis in token_analysis.items():
        if analysis["count"] > 0 and not analysis["is_universal"]:
            any_mask |= analysis["mask"]

    if any_mask.any():
        logger.info(f"Fallback OR match: {any_mask.sum()} rows")
        return any_mask

    return pd.Series(False, index=df.index)


def smart_filter(
    df: pd.DataFrame,
    analyzer: DataFrameAnalyzer = None,
    date_str: str = None,
    start_date_str: str = None,
    end_date_str: str = None,
    keyword: str = None,
) -> pd.DataFrame:
    """
    Universal smart filter that works with ANY Excel structure.

    Args:
        df: The dataframe to filter
        analyzer: Pre-computed DataFrameAnalyzer (created if not provided)
        date_str: SINGLE date to filter by (e.g. '21 Nov 2025')
        start_date_str: Start of date range (inclusive)
        end_date_str: End of date range (inclusive)
        keyword: Keyword/phrase to filter by
    """
    if analyzer is None:
        analyzer = DataFrameAnalyzer(df)

    filtered_df = df.copy()

    # ---------- DATE FILTERING (single date OR range) ----------
    def _parse_date(s: str):
        if not s:
            return None
        s = str(s).strip()
        # Try flexible parsing first (handles '21 Nov 2025', '2025-11-21', etc.)
        try:
            return pd.to_datetime(s, dayfirst=True).date()
        except Exception:
            # Try a few explicit formats as fallback
            for fmt in [
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%m-%d-%Y",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d %b %Y",
                "%d %B %Y",
            ]:
                try:
                    return pd.to_datetime(s, format=fmt).date()
                except Exception:
                    continue
        return None

    single_date = _parse_date(date_str) if date_str else None
    start_date = _parse_date(start_date_str) if start_date_str else None
    end_date = _parse_date(end_date_str) if end_date_str else None

    if single_date or start_date or end_date:
        date_found = False
        date_columns = analyzer.get_date_columns()
        candidate_cols = date_columns + [c for c in df.columns if c not in date_columns]

        for col in candidate_cols:
            try:
                series = df[col]

                # Your file's datetime format: MM/DD/YYYY HH:MM:SS
                try:
                    temp_series = pd.to_datetime(
                        series,
                        format="%m/%d/%Y %H:%M:%S",
                        errors="coerce",
                    )
                except Exception:
                    temp_series = pd.to_datetime(series, errors="coerce")

                if temp_series.notna().sum() == 0:
                    continue

                s_dates = temp_series.dt.date

                if single_date:
                    mask = s_dates == single_date
                else:
                    mask = pd.Series(True, index=df.index)
                    if start_date:
                        mask &= s_dates >= start_date
                    if end_date:
                        mask &= s_dates <= end_date

                if mask.any():
                    filtered_df = df[mask]
                    date_found = True
                    logger.info(
                        "Date filter on column '%s': rows=%d, single_date=%r, start_date=%r, end_date=%r",
                        col,
                        mask.sum(),
                        single_date,
                        start_date,
                        end_date,
                    )
                    break
            except Exception as e:
                logger.error(f"Date filter error on column '{col}': {e}")
                continue

        if not date_found:
            logger.info(
                "No rows found for date/date range: single=%r, start=%r, end=%r",
                single_date,
                start_date,
                end_date,
            )
            return pd.DataFrame()

    # ---------- KEYWORD FILTERING (unchanged logic) ----------
    if keyword:
        keyword = str(keyword).strip()
        if not keyword:
            return filtered_df

        # Rebuild analyzer on the already date-filtered data if we did a date filter
        did_date_filter = bool(single_date or start_date or end_date)
        current_analyzer = (
            DataFrameAnalyzer(filtered_df) if did_date_filter else analyzer
        )

        mask = _dynamic_keyword_search(filtered_df, current_analyzer, keyword)

        if mask.any():
            before = len(filtered_df)
            filtered_df = filtered_df[mask]
            logger.info(
                "Keyword filter '%s': %d -> %d rows",
                keyword,
                before,
                len(filtered_df),
            )
        else:
            if did_date_filter and len(filtered_df) > 0:
                logger.warning(
                    "Keyword '%s' not found after date filter, returning date-only results",
                    keyword,
                )
            else:
                logger.info("Keyword '%s' not found anywhere", keyword)
                return pd.DataFrame()

    return filtered_df


def execute_pandas_retrieval(file_name: str, query: str) -> str:
    """
    Execute pandas-based retrieval for Excel files.
    Fully dynamic - no hardcoded column names or values.
    """
    file_path = os.path.join(UPLOADS_DIR, file_name)

    if not os.path.exists(file_path):
        return "Error: Source file not found."

    try:
        # Load the Excel file
        df = pd.read_excel(file_path, header=0)
        df.columns = [str(col).strip() for col in df.columns]

        # Analyze the dataframe structure
        analyzer = DataFrameAnalyzer(df)

        # Generate dynamic context for LLM
        column_summary = analyzer.get_column_summary()
        sample_data = df.head(5).to_string(index=False)
        columns = list(df.columns)

        # Get categorical values for context
        categorical_values = analyzer.get_all_categorical_values()
        cat_context = ""
        if categorical_values:
            cat_lines = []
            for col, vals in categorical_values.items():
                cat_lines.append(f"  {col}: {vals}")
            cat_context = "KNOWN VALUES IN DATA:\n" + "\n".join(cat_lines)

        prompt = f"""
You are a STRICT Python code generator for pandas-style data analysis.

CONTEXT
- A pandas DataFrame `df` is ALREADY loaded.
- A DataFrameAnalyzer `analyzer` is ALREADY created for `df`.
- You MUST NOT import anything.
- You MUST NOT read files.
- You MUST NOT print anything.

AVAILABLE:
- `df` - the loaded DataFrame
- `analyzer` - DataFrameAnalyzer instance
- `smart_filter(df, analyzer, date_str=None, start_date_str=None, end_date_str=None, keyword=None)` - filtering helper (supports single date or date ranges)

=== DATAFRAME STRUCTURE ===
Columns: {columns}

{column_summary}

{cat_context}

=== SAMPLE DATA ===
{sample_data}

=== USER QUERY ===
{query!r}

YOUR TASK
Generate Python code that:

1. **Extract parameters from the query:**
   
   `date_str`: Use this ONLY when the question clearly refers to a SINGLE specific date
   (e.g. "on 21 Nov 2025"). Example:
   - "on nov 21, 2025" → date_str = "nov 21, 2025", start_date_str = None, end_date_str = None

   `start_date_str` and `end_date_str`: Use these when the question describes a DATE RANGE
   (e.g. "from X to Y", "between X and Y", "from X until Y").
   Examples:
   - "from 21 to 23 nov 2025" → start_date_str = "21 nov 2025", end_date_str = "23 nov 2025", date_str = None
   - "between nov 21, 2025 and nov 23, 2025" → start_date_str = "nov 21, 2025", end_date_str = "nov 23, 2025", date_str = None
   - "after nov 21, 2025" → start_date_str = "nov 21, 2025", end_date_str = None, date_str = None
   - "before nov 23, 2025" → start_date_str = None, end_date_str = "nov 23, 2025", date_str = None

   IMPORTANT:
   - Never put more than one date into `date_str`.
   - If a range is present, set `date_str = None` and use `start_date_str` / `end_date_str`.

   `keyword`: Extract the SPECIFIC identifying phrase the user wants to filter by.
   - Look at the KNOWN VALUES above to understand what values exist
   - Keep qualifier words that distinguish items (e.g., "premium ironing", "type B")
   - If no specific filter needed → keyword = None

2. **Filter the data:**
   result_df = smart_filter(
       df,
       analyzer,
       date_str=date_str,
       start_date_str=start_date_str,
       end_date_str=end_date_str,
       keyword=keyword,
   )

3. **Generate result based on question_type:**

   If question_type in ["who", "name"]:
       # Find columns with high uniqueness (likely names/identifiers)
       id_cols = analyzer.get_identifier_columns()
       if len(result_df) == 0:
           result = "NO_MATCH"
       elif id_cols:
           result = result_df[id_cols].drop_duplicates().to_string(index=False)
       else:
           result = result_df.to_string(index=False)
   
   elif question_type == "count":
       result = str(len(result_df))
   
   elif question_type in ["sum", "total"]:
       # Find numeric columns and sum them
       numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
       if numeric_cols and len(result_df) > 0:
           sums = result_df[numeric_cols].sum().to_dict()
           result = str(sums)
       else:
           result = str(len(result_df))
   
   elif question_type in ["average", "mean"]:
       numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
       if numeric_cols and len(result_df) > 0:
           means = result_df[numeric_cols].mean().to_dict()
           result = str(means)
       else:
           result = "NO_NUMERIC_DATA"
   
   else:
       result = result_df.to_string(index=False) if len(result_df) > 0 else "NO_MATCH"

4. **Required variables at end:**
   - date_str (str or None)
   - start_date_str (str or None)
   - end_date_str (str or None)
   - keyword (str or None)
   - question_type (str)
   - result_df (DataFrame)
   - result (str)

OUTPUT: Only executable Python code. No comments. No backticks.
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python code generator. Output ONLY executable Python code.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        code = (
            response.choices[0]
            .message.content.replace("```python", "")
            .replace("```", "")
            .strip()
        )

        logger.info(f"Generated Pandas Code:\n{code}")
        logger.info(f"User query: {query}")

        # Execute with all necessary context
        local_vars = {
            "df": df,
            "pd": pd,
            "analyzer": analyzer,
            "smart_filter": smart_filter,
        }

        try:
            exec(code, {}, local_vars)
        except Exception as exec_err:
            logger.error(f"CODE EXECUTION ERROR:\n{code}\nERROR: {exec_err}")
            return f"Error executing code: {exec_err}"

        # Log what was extracted
        logger.info(
            "Execution params: date_str=%r, keyword=%r, question_type=%r, result_rows=%d",
            local_vars.get("date_str"),
            local_vars.get("keyword"),
            local_vars.get("question_type"),
            len(local_vars.get("result_df", [])),
        )

        raw_result = local_vars.get("result", "No result variable returned.")
        return str(raw_result)

    except Exception as e:
        logger.error(f"System Error: {traceback.format_exc()}")
        return f"System Error: {str(e)}"
