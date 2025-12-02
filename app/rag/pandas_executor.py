# pandas_executor.py
import os
import re
import math
import pandas as pd
import numpy as np
import traceback
from typing import Optional, List, Dict, Any, Tuple
from app.core.openai_client import client, CHAT_MODEL
from app.core.logger import logger

UPLOADS_DIR = os.path.abspath("./uploads")

# Your fixed Excel date format
EXCEL_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


# ============================================================================
# JSON SANITIZATION HELPERS
# ============================================================================


def sanitize_for_json(obj):
    """
    Recursively sanitize an object for JSON serialization.
    Converts NaN, Infinity, -Infinity to None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    elif pd.isna(obj):
        return None
    else:
        return obj


def dataframe_to_json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to list of dicts, handling NaN values properly.
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Convert timestamps to strings
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

    # Replace NaN with None
    df_clean = df_clean.replace({np.nan: None})

    # Handle any remaining NaT values
    df_clean = df_clean.where(pd.notnull(df_clean), None)

    # Convert to records
    records = df_clean.to_dict(orient="records")

    # Final sanitization pass
    return sanitize_for_json(records)


# ============================================================================
# QUERY RESULT CLASS
# ============================================================================


class QueryResult:
    """
    Enhanced result class that distinguishes between different response types.
    """

    def __init__(self):
        self.success: bool = False
        self.response_type: str = "text"  # "text", "table", "count", "value"
        self.intro_message: str = ""
        self.raw_data: Optional[pd.DataFrame] = None
        self.scalar_value: Optional[str] = None
        self.debug: Dict[str, Any] = {}
        self.total_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response_type": self.response_type,
            "intro_message": self.intro_message,
            "raw_data": self.raw_data,
            "scalar_value": self.scalar_value,
            "total_rows": self.total_rows,
            "debug": self.debug,
        }


# ============================================================================
# DYNAMIC QUERY ENGINE
# ============================================================================


class DynamicQueryEngine:
    """
    A fully dynamic query engine that learns everything from the data.
    No hardcoded mappings or assumptions about column names/values.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = list(df.columns)
        self.column_profiles = self._profile_all_columns()

    def _profile_all_columns(self) -> Dict[str, Dict]:
        """
        Profile every column to understand its characteristics.
        Everything is learned from the data itself.
        """
        profiles = {}

        for col in self.columns:
            series = self.df[col]
            non_null = series.dropna()

            profile = {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "total_count": len(series),
                "unique_count": int(series.nunique()),
                "unique_ratio": float(series.nunique() / max(len(series), 1)),
            }

            # Detect if numeric
            if pd.api.types.is_numeric_dtype(series):
                profile["type"] = "numeric"
                if len(non_null) > 0:
                    profile["min"] = (
                        float(series.min()) if not pd.isna(series.min()) else None
                    )
                    profile["max"] = (
                        float(series.max()) if not pd.isna(series.max()) else None
                    )
                    profile["mean"] = (
                        float(series.mean()) if not pd.isna(series.mean()) else None
                    )
                else:
                    profile["min"] = None
                    profile["max"] = None
                    profile["mean"] = None

            # Detect if datetime
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile["type"] = "datetime"

            # For object/string columns
            else:
                # Try to detect if it's a date column by parsing
                is_date = self._detect_date_column(non_null)

                if is_date:
                    profile["type"] = "datetime"
                else:
                    profile["type"] = "text"

                    # Store unique values if cardinality is low (categorical-like)
                    if profile["unique_count"] <= 100:
                        profile["unique_values"] = [
                            str(v) for v in non_null.unique().tolist()
                        ]

                    # Store sample values for high cardinality columns
                    profile["sample_values"] = [
                        str(v) for v in non_null.head(20).tolist()
                    ]

            profiles[col] = profile

        return profiles

    def _detect_date_column(self, series: pd.Series) -> bool:
        """Detect if a series contains date values using the known format."""
        if len(series) == 0:
            return False

        try:
            sample = series.head(20).astype(str)
            parsed = pd.to_datetime(sample, format=EXCEL_DATE_FORMAT, errors="coerce")
            valid_ratio = parsed.notna().sum() / len(sample)
            return valid_ratio > 0.7
        except Exception:
            return False

    def find_value_in_dataframe(self, search_term: str) -> List[Dict]:
        """
        Search for a term across ALL columns and return where it's found.
        Completely data-driven - no assumptions.
        """
        search_lower = str(search_term).lower().strip()
        findings = []

        for col in self.columns:
            profile = self.column_profiles[col]

            # Skip numeric and date columns for text search
            if profile["type"] in ["numeric", "datetime"]:
                continue

            series = self.df[col].astype(str).str.lower()

            # Check for exact matches
            exact_mask = series == search_lower
            exact_count = exact_mask.sum()

            # Check for partial/contains matches
            try:
                contains_mask = series.str.contains(re.escape(search_lower), na=False)
                contains_count = contains_mask.sum()
            except Exception:
                contains_mask = pd.Series(False, index=self.df.index)
                contains_count = 0

            if contains_count > 0:
                # Get the actual matched values
                matched_rows = self.df[contains_mask]
                matched_values = matched_rows[col].unique()[:5]

                findings.append(
                    {
                        "column": col,
                        "exact_matches": int(exact_count),
                        "contains_matches": int(contains_count),
                        "selectivity": float(contains_count / len(self.df)),
                        "matched_values": [str(v) for v in matched_values],
                        "mask": contains_mask,
                    }
                )

        # Sort by selectivity (most selective first)
        findings.sort(key=lambda x: (x["selectivity"], -x["exact_matches"]))

        return findings

    def find_number_in_dataframe(
        self, number: float, tolerance: float = 0.01
    ) -> List[Dict]:
        """
        Search for a numeric value across ALL numeric columns.
        """
        findings = []

        for col in self.columns:
            profile = self.column_profiles[col]

            if profile["type"] != "numeric":
                continue

            try:
                series = self.df[col]
                mask = (series - number).abs() <= tolerance

                if mask.sum() > 0:
                    findings.append(
                        {
                            "column": col,
                            "match_count": int(mask.sum()),
                            "selectivity": float(mask.sum() / len(self.df)),
                            "mask": mask,
                        }
                    )
            except Exception:
                continue

        findings.sort(key=lambda x: x["selectivity"])

        return findings

    def find_date_in_dataframe(self, target_date) -> List[Dict]:
        """
        Search for a date across ALL date columns.
        """
        findings = []

        for col in self.columns:
            profile = self.column_profiles[col]

            if profile["type"] != "datetime":
                continue

            try:
                parsed = pd.to_datetime(
                    self.df[col], format=EXCEL_DATE_FORMAT, errors="coerce"
                )

                mask = parsed.dt.date == target_date

                if mask.sum() > 0:
                    findings.append(
                        {
                            "column": col,
                            "match_count": int(mask.sum()),
                            "selectivity": float(mask.sum() / len(self.df)),
                            "mask": mask,
                        }
                    )
            except Exception as e:
                logger.debug(f"Date search error on column {col}: {e}")
                continue

        findings.sort(key=lambda x: x["selectivity"])

        return findings

    def get_data_summary_for_llm(self) -> str:
        """
        Generate a comprehensive summary of the data for the LLM.
        """
        lines = [
            "=== DATA STRUCTURE ===",
            f"Total Rows: {len(self.df)}",
            f"Columns: {len(self.columns)}",
            "",
        ]

        for col, profile in self.column_profiles.items():
            line = f"[{col}] Type: {profile['type']}, Unique: {profile['unique_count']}"

            if profile["type"] == "numeric":
                line += f", Range: {profile.get('min')} to {profile.get('max')}"

            if profile.get("unique_values"):
                vals_preview = profile["unique_values"][:7]
                line += f", Values: {vals_preview}"

            if profile.get("sample_values") and not profile.get("unique_values"):
                samples = profile["sample_values"][:5]
                line += f", Samples: {samples}"

            lines.append(line)

        return "\n".join(lines)


# ============================================================================
# QUERY TYPE DETECTION
# ============================================================================


def detect_list_query(query: str) -> bool:
    """
    Detect if the user is asking for a list/table of data.
    Returns True if it's a list query.
    """
    query_lower = query.lower()

    # Patterns that indicate list/table requests
    list_patterns = [
        r"\blist\b",
        r"\bshow\s+(me\s+)?(all|the)\b",
        r"\bshow\s+all\b",
        r"\bdisplay\b",
        r"\bget\s+(me\s+)?(all|the)\b",
        r"\bfetch\s+(all|the)\b",
        r"\bgive\s+(me\s+)?(all|the|a\s+list)\b",
        r"\bwhat\s+are\s+(all\s+)?(the\s+)?",
        r"\bwhich\s+(all\s+)?",
        r"\bfind\s+(all|the)\b",
        r"\ball\s+(the\s+)?\w+\s+(with|where|having|from|in)\b",
        r"\beveryone\b",
        r"\ball\s+(orders|records|entries|items|customers|users|products|transactions)\b",
        r"\bdetails\s+of\s+(all|the)\b",
        r"\bentire\s+list\b",
        r"\bcomplete\s+list\b",
        r"\bexport\b",
        r"\btable\s+of\b",
        r"\bgive\s+me\s+the\s+details\b",
        r"\bdetails\s+of\b",
    ]

    for pattern in list_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


def detect_query_type(query: str, components: Dict[str, Any]) -> str:
    """
    Determine the type of query based on question and components.
    Returns: "list", "count", "sum", "specific_value", "filter_only"
    """
    query_lower = query.lower()

    # Check for count queries
    count_patterns = [
        r"\bhow\s+many\b",
        r"\bcount\s+(of|the)?\b",
        r"\bnumber\s+of\b",
        r"\btotal\s+count\b",
    ]
    for pattern in count_patterns:
        if re.search(pattern, query_lower):
            return "count"

    # Check for sum/total queries
    sum_patterns = [
        r"\btotal\s+(amount|value|sum|price|cost)\b",
        r"\bsum\s+of\b",
        r"\baggregate\b",
        r"\bcombined\s+(total|amount)\b",
    ]
    for pattern in sum_patterns:
        if re.search(pattern, query_lower):
            return "sum"

    # Check for list queries
    if detect_list_query(query):
        return "list"

    # Check for specific value queries (what is the X of Y)
    specific_patterns = [
        r"\bwhat\s+is\s+(the\s+)?\w+\s+(of|for)\b",
        r"\btell\s+me\s+(the\s+)?\w+\s+(of|for)\b",
        r"\bwhat\'s\s+(the\s+)?\b",
    ]
    for pattern in specific_patterns:
        if re.search(pattern, query_lower):
            return "specific_value"

    # Use LLM-extracted type if available
    llm_type = components.get("question_type", "list")

    return llm_type


# ============================================================================
# LLM HELPERS
# ============================================================================


def extract_query_components_with_llm(
    query: str, engine: DynamicQueryEngine
) -> Dict[str, Any]:
    """
    Use LLM to understand the query in context of the actual data.
    """
    data_summary = engine.get_data_summary_for_llm()
    sample_data = engine.df.head(3).to_string(index=False)

    prompt = f"""You are analyzing a user's question about data in an Excel file.

USER QUESTION: "{query}"

ACTUAL DATA STRUCTURE:
{data_summary}

SAMPLE DATA (first 3 rows):
{sample_data}

YOUR TASK:
Extract the components from the user's question that can be used to filter and query this specific data.

Respond with a JSON object containing:

1. "search_terms": List of text values to search for (names, identifiers, categories mentioned in the question)
   - Extract ANY word/phrase that might match data in text columns
   - Look at the sample data and column values above to identify what the user might be referring to

2. "date_value": A date mentioned in the question, converted to "YYYY-MM-DD" format, or null if no date
   - Handle formats like "14th nov 2025", "november 14, 2025", "17th November 5:56pm", etc.

3. "time_value": A specific time mentioned (like "5:56pm"), in "HH:MM:SS" 24-hour format, or null
   - Convert "5:56pm" to "17:56:00"

4. "numeric_values": List of numbers mentioned (excluding years), or empty list
   - Include amounts, quantities, prices, etc.

5. "target_attribute": The specific piece of information the user wants to know
   - This should be a description like "payment method", "status", "total amount", etc.
   - Or null if they want all information / a list

6. "question_type": One of "specific_value", "count", "sum", "list", "filter_only"
   - "specific_value": User wants one specific attribute (e.g., "what is the payment mode")
   - "count": User wants a count (e.g., "how many orders")
   - "sum": User wants a total (e.g., "total amount")
   - "list": User wants to see matching records or details
   - "filter_only": User just wants filtered data

7. "columns_to_display": List of column names the user wants to see, or null for all columns
   - If user says "show names and emails", extract ["name", "email"] or similar column names from the data

IMPORTANT:
- Base your extraction on the ACTUAL column names and values shown above
- The search_terms should be things that would actually appear in the data
- Be precise - extract exactly what's in the question
- "give me the details" means user wants ALL columns (list type)

Respond ONLY with valid JSON, no other text:
"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You extract query components from natural language questions. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        if "```" in response_text:
            response_text = re.sub(r"```json?\s*", "", response_text)
            response_text = re.sub(r"```\s*$", "", response_text)

        import json

        parsed = json.loads(response_text)

        logger.info(f"LLM extracted components: {parsed}")
        return parsed

    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {
            "search_terms": [],
            "date_value": None,
            "time_value": None,
            "numeric_values": [],
            "target_attribute": None,
            "question_type": "list",
            "columns_to_display": None,
        }


def generate_intro_message(
    query: str, row_count: int, filter_description: str = ""
) -> str:
    """
    Generate a brief introductory message for list responses.
    This is a SHORT message, not the data itself.
    """
    try:
        prompt = f"""Generate a brief, friendly one-sentence introduction for displaying data results.

User's question: "{query}"
Number of records found: {row_count}
Filters applied: {filter_description if filter_description else "None"}

Generate ONLY a brief intro sentence like:
- "Here are the 5 orders from November 2024:"
- "Found 12 customers matching your criteria:"
- "Here's the complete list of pending transactions:"
- "Here are the details of the order closed on November 17th:"

Keep it under 15 words. End with a colon. Do not include the actual data.
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You generate brief introductory sentences. Keep responses under 15 words.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=50,
        )

        intro = response.choices[0].message.content.strip()

        # Ensure it ends with a colon
        if not intro.endswith(":"):
            intro = intro.rstrip(".") + ":"

        return intro

    except Exception as e:
        logger.error(f"Intro generation failed: {e}")
        return f"Found {row_count} matching records:"


def find_target_column_with_llm(
    query: str, target_description: str, engine: DynamicQueryEngine
) -> Optional[str]:
    """
    Use LLM to identify which column contains the target attribute.
    """
    if not target_description:
        return None

    columns_info = []
    for col, profile in engine.column_profiles.items():
        info = f"- {col}: {profile['type']}"
        if profile.get("unique_values"):
            info += f", values like: {profile['unique_values'][:5]}"
        elif profile.get("sample_values"):
            info += f", samples: {profile['sample_values'][:3]}"
        columns_info.append(info)

    prompt = f"""Given this user question and target attribute, identify the exact column name that contains the answer.

USER QUESTION: "{query}"
TARGET ATTRIBUTE: "{target_description}"

AVAILABLE COLUMNS:
{chr(10).join(columns_info)}

Which column name contains the "{target_description}"?
Respond with ONLY the exact column name from the list above, or "NONE" if no column matches.
"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You identify column names. Respond with only the column name or NONE.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content.strip()

        if result in engine.columns:
            return result

        for col in engine.columns:
            if col.lower() == result.lower():
                return col

        return None

    except Exception as e:
        logger.error(f"Target column identification failed: {e}")
        return None


# ============================================================================
# DATA FORMATTING
# ============================================================================


def dataframe_to_markdown(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """
    Convert a DataFrame to a clean Markdown table.
    """
    if df.empty:
        return "No data found."

    display_df = df.head(max_rows) if max_rows else df

    # Build markdown table
    headers = list(display_df.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    rows = []
    for _, row in display_df.iterrows():
        row_values = []
        for v in row:
            if pd.isna(v):
                row_values.append("")
            elif isinstance(v, pd.Timestamp):
                row_values.append(v.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                row_values.append(str(v))
        row_str = "| " + " | ".join(row_values) + " |"
        rows.append(row_str)

    markdown = "\n".join([header_row, separator] + rows)

    if max_rows and len(df) > max_rows:
        markdown += f"\n\n*... and {len(df) - max_rows} more rows*"

    return markdown


def dataframe_to_html(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """
    Convert a DataFrame to an HTML table with styling.
    """
    if df.empty:
        return "<p>No data found.</p>"

    display_df = df.head(max_rows) if max_rows else df

    # Clean the dataframe for HTML
    display_df_clean = display_df.copy()
    for col in display_df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df_clean[col]):
            display_df_clean[col] = display_df_clean[col].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
            )

    html = display_df_clean.to_html(
        index=False, classes="data-table", border=0, na_rep=""
    )

    # Add basic styling
    styled_html = f"""
<style>
.data-table {{
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
    font-size: 14px;
}}
.data-table th {{
    background-color: #4CAF50;
    color: white;
    padding: 12px 8px;
    text-align: left;
    border-bottom: 2px solid #ddd;
}}
.data-table td {{
    padding: 10px 8px;
    border-bottom: 1px solid #ddd;
}}
.data-table tr:nth-child(even) {{
    background-color: #f9f9f9;
}}
.data-table tr:hover {{
    background-color: #f1f1f1;
}}
</style>
{html}
"""

    if max_rows and len(df) > max_rows:
        styled_html += f"<p><em>... and {len(df) - max_rows} more rows</em></p>"

    return styled_html


# ============================================================================
# QUERY EXECUTION
# ============================================================================


def execute_dynamic_query(
    engine: DynamicQueryEngine, components: Dict[str, Any], query: str
) -> QueryResult:
    """
    Execute the query using extracted components.
    Returns a QueryResult with proper type distinction.
    """
    result = QueryResult()
    result.debug = {
        "components": components,
        "filters_applied": [],
        "rows_progression": [len(engine.df)],
    }

    combined_mask = pd.Series(True, index=engine.df.index)
    filter_descriptions = []

    # 1. Apply text search filters
    search_terms = components.get("search_terms", [])
    for term in search_terms:
        if not term or len(str(term).strip()) < 2:
            continue

        findings = engine.find_value_in_dataframe(str(term))

        if findings:
            best_finding = findings[0]

            if best_finding["selectivity"] < 0.8:
                combined_mask &= best_finding["mask"]
                filter_descriptions.append(f"{best_finding['column']}='{term}'")
                result.debug["filters_applied"].append(
                    {
                        "type": "text_search",
                        "term": term,
                        "column": best_finding["column"],
                        "matches": best_finding["contains_matches"],
                    }
                )

                temp_count = combined_mask.sum()
                result.debug["rows_progression"].append(int(temp_count))

                if temp_count == 0:
                    combined_mask |= best_finding["mask"]
                    filter_descriptions.pop()

    # 2. Apply date filter
    date_value = components.get("date_value")
    if date_value:
        try:
            target_date = pd.to_datetime(date_value).date()
            date_findings = engine.find_date_in_dataframe(target_date)

            if date_findings:
                best_date = date_findings[0]
                new_mask = combined_mask & best_date["mask"]

                if new_mask.sum() > 0:
                    combined_mask = new_mask
                    filter_descriptions.append(f"{best_date['column']}='{date_value}'")
                    result.debug["filters_applied"].append(
                        {
                            "type": "date_filter",
                            "date": date_value,
                            "column": best_date["column"],
                        }
                    )
                    result.debug["rows_progression"].append(int(combined_mask.sum()))

        except Exception as e:
            logger.error(f"Date parsing error: {e}")

    # 3. Apply time filter if specified
    time_value = components.get("time_value")
    if time_value and date_value:
        try:
            # Find datetime columns and filter by time
            for col in engine.columns:
                profile = engine.column_profiles[col]
                if profile["type"] != "datetime":
                    continue

                parsed = pd.to_datetime(
                    engine.df[col], format=EXCEL_DATE_FORMAT, errors="coerce"
                )

                # Parse target time
                target_time = pd.to_datetime(time_value).time()

                # Create time mask with some tolerance (within 1 minute)
                time_mask = (parsed.dt.hour == target_time.hour) & (
                    parsed.dt.minute == target_time.minute
                )

                new_mask = combined_mask & time_mask
                if new_mask.sum() > 0:
                    combined_mask = new_mask
                    filter_descriptions.append(f"time='{time_value}'")
                    result.debug["filters_applied"].append(
                        {
                            "type": "time_filter",
                            "time": time_value,
                            "column": col,
                        }
                    )
                    result.debug["rows_progression"].append(int(combined_mask.sum()))
                    break

        except Exception as e:
            logger.debug(f"Time filter error: {e}")

    # 4. Apply numeric filter
    numeric_values = components.get("numeric_values", [])
    if numeric_values and combined_mask.sum() > 1:
        for num in numeric_values:
            try:
                num_findings = engine.find_number_in_dataframe(float(num))

                if num_findings:
                    best_num = num_findings[0]

                    if best_num["selectivity"] < 0.5:
                        new_mask = combined_mask & best_num["mask"]

                        if new_mask.sum() > 0:
                            combined_mask = new_mask
                            filter_descriptions.append(f"{best_num['column']}={num}")
                            result.debug["filters_applied"].append(
                                {
                                    "type": "numeric_filter",
                                    "value": num,
                                    "column": best_num["column"],
                                }
                            )
                            result.debug["rows_progression"].append(
                                int(combined_mask.sum())
                            )

            except Exception as e:
                logger.debug(f"Numeric filter error: {e}")

    # 5. Get filtered dataframe
    filtered_df = engine.df[combined_mask]
    result.debug["final_row_count"] = len(filtered_df)
    result.total_rows = len(filtered_df)

    if filtered_df.empty:
        result.success = False
        result.response_type = "text"
        result.scalar_value = "NO_MATCH"
        return result

    # 6. Determine query type and handle accordingly
    query_type = detect_query_type(query, components)

    filter_desc = ", ".join(filter_descriptions) if filter_descriptions else ""

    # Handle COUNT queries
    if query_type == "count":
        result.success = True
        result.response_type = "count"
        result.scalar_value = str(len(filtered_df))
        result.intro_message = f"The count is {len(filtered_df)}."
        return result

    # Handle SUM queries
    if query_type == "sum":
        target_description = components.get("target_attribute")
        if target_description:
            target_col = find_target_column_with_llm(query, target_description, engine)
            if target_col and engine.column_profiles[target_col]["type"] == "numeric":
                total = filtered_df[target_col].sum()
                result.success = True
                result.response_type = "value"
                result.scalar_value = str(total)
                result.intro_message = f"The total {target_description} is {total}."
                return result

    # Handle SPECIFIC VALUE queries
    if query_type == "specific_value":
        target_description = components.get("target_attribute")
        if target_description:
            target_col = find_target_column_with_llm(query, target_description, engine)

            if target_col and target_col in filtered_df.columns:
                values = filtered_df[target_col].dropna().unique()

                if len(values) == 1:
                    result.success = True
                    result.response_type = "value"
                    val = values[0]
                    result.scalar_value = str(val) if not pd.isna(val) else "N/A"
                    return result
                elif len(values) > 1 and len(values) <= 5:
                    result.success = True
                    result.response_type = "value"
                    result.scalar_value = ", ".join(
                        str(v) for v in values if not pd.isna(v)
                    )
                    return result
                # If many values, fall through to list handling

    # Handle LIST queries (default for multiple results or "give me details")
    if query_type == "list" or len(filtered_df) >= 1:
        result.success = True
        result.response_type = "table"

        # Select columns if specified
        columns_to_display = components.get("columns_to_display")
        if columns_to_display:
            # Match requested columns to actual columns
            matched_cols = []
            for req_col in columns_to_display:
                for actual_col in filtered_df.columns:
                    if (
                        req_col.lower() in actual_col.lower()
                        or actual_col.lower() in req_col.lower()
                    ):
                        if actual_col not in matched_cols:
                            matched_cols.append(actual_col)
                        break

            if matched_cols:
                filtered_df = filtered_df[matched_cols]

        result.raw_data = filtered_df
        result.intro_message = generate_intro_message(
            query, len(filtered_df), filter_desc
        )
        return result

    # Default: return as table
    result.success = True
    result.response_type = "table"
    result.raw_data = filtered_df
    result.intro_message = generate_intro_message(query, len(filtered_df), filter_desc)
    return result


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def execute_pandas_retrieval(file_name: str, query: str) -> Dict[str, Any]:
    """
    Main entry point for dynamic pandas-based data retrieval.
    Returns a structured result that can be handled by the route.
    """
    file_path = os.path.join(UPLOADS_DIR, file_name)

    if not os.path.exists(file_path):
        return {
            "success": False,
            "response_type": "error",
            "answer": "Error: Source file not found.",
        }

    try:
        # Load Excel file
        df = pd.read_excel(file_path, header=0)
        df.columns = [str(col).strip() for col in df.columns]

        logger.info(f"Loaded file: {file_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Initialize dynamic query engine
        engine = DynamicQueryEngine(df)

        # Step 1: Extract query components using LLM
        components = extract_query_components_with_llm(query, engine)

        # Step 2: Execute the query dynamically
        result = execute_dynamic_query(engine, components, query)

        logger.info(
            f"Query result: success={result.success}, type={result.response_type}"
        )

        # Step 3: Format the response based on type
        if result.response_type == "table" and result.raw_data is not None:
            # Use JSON-safe conversion for table_data
            table_data = dataframe_to_json_safe_records(result.raw_data)

            return {
                "success": True,
                "response_type": "table",
                "intro_message": result.intro_message,
                "table_data": table_data,  # JSON-safe
                "table_markdown": dataframe_to_markdown(result.raw_data),
                "table_html": dataframe_to_html(result.raw_data),
                "columns": list(result.raw_data.columns),
                "total_rows": len(result.raw_data),
                "debug": sanitize_for_json(result.debug),
            }

        elif result.response_type in ["count", "value"]:
            return {
                "success": True,
                "response_type": result.response_type,
                "answer": result.scalar_value,
                "intro_message": result.intro_message,
                "debug": sanitize_for_json(result.debug),
            }

        else:
            # Fallback or NO_MATCH
            if result.scalar_value == "NO_MATCH":
                # Try alternative approach
                alt_result = _direct_llm_query(df, engine, query)
                if alt_result and alt_result.get("success"):
                    return alt_result

            return {
                "success": False,
                "response_type": "text",
                "answer": result.scalar_value or "NO_MATCH",
                "debug": sanitize_for_json(result.debug),
            }

    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return {
            "success": False,
            "response_type": "error",
            "answer": f"Error: {str(e)}",
        }


def _direct_llm_query(
    df: pd.DataFrame, engine: DynamicQueryEngine, query: str
) -> Dict[str, Any]:
    """
    Fallback: Let LLM generate specific filter conditions based on actual data.
    """
    try:
        data_summary = engine.get_data_summary_for_llm()
        sample_rows = df.head(5).to_string(index=False)

        prompt = f"""Analyze this question and the actual data to find the answer.

QUESTION: "{query}"

DATA STRUCTURE:
{data_summary}

SAMPLE DATA:
{sample_rows}

Based on the actual data shown above:

1. Identify which column(s) need to be filtered and with what values
2. Identify which column contains the answer

Respond with JSON:
{{
    "filters": [
        {{"column": "exact_column_name_from_data", "value": "exact_value_to_search", "match_type": "exact|contains"}}
    ],
    "answer_column": "exact_column_name_containing_answer",
    "return_all_columns": true/false,
    "reasoning": "brief explanation"
}}

Use EXACT column names and realistic values based on the data shown.
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You analyze data and extract filter conditions. Respond only with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        if "```" in response_text:
            response_text = re.sub(r"```json?\s*", "", response_text)
            response_text = re.sub(r"```\s*$", "", response_text)

        import json

        parsed = json.loads(response_text)

        logger.info(f"Direct LLM query plan: {parsed}")

        # Apply filters
        filtered_df = df.copy()

        for f in parsed.get("filters", []):
            col = f.get("column")
            val = f.get("value")
            match_type = f.get("match_type", "contains")

            if col not in filtered_df.columns:
                continue

            col_profile = engine.column_profiles.get(col, {})

            if col_profile.get("type") == "datetime":
                try:
                    parsed_col = pd.to_datetime(
                        filtered_df[col], format=EXCEL_DATE_FORMAT, errors="coerce"
                    )
                    target_date = pd.to_datetime(val).date()
                    mask = parsed_col.dt.date == target_date
                    new_df = filtered_df[mask]
                    if len(new_df) > 0:
                        filtered_df = new_df
                except Exception:
                    pass

            elif col_profile.get("type") == "numeric":
                try:
                    num_val = float(val)
                    mask = (filtered_df[col] - num_val).abs() < 0.01
                    new_df = filtered_df[mask]
                    if len(new_df) > 0:
                        filtered_df = new_df
                except Exception:
                    pass

            else:
                val_lower = str(val).lower()
                if match_type == "exact":
                    mask = filtered_df[col].astype(str).str.lower() == val_lower
                else:
                    mask = (
                        filtered_df[col]
                        .astype(str)
                        .str.lower()
                        .str.contains(re.escape(val_lower), na=False)
                    )

                new_df = filtered_df[mask]
                if len(new_df) > 0:
                    filtered_df = new_df

        if filtered_df.empty:
            return {"success": False, "response_type": "text", "answer": "NO_MATCH"}

        # Check if list or specific value
        is_list_query = detect_list_query(query) or parsed.get(
            "return_all_columns", False
        )

        if is_list_query or len(filtered_df) > 1:
            # Use JSON-safe conversion
            table_data = dataframe_to_json_safe_records(filtered_df)

            return {
                "success": True,
                "response_type": "table",
                "intro_message": generate_intro_message(query, len(filtered_df)),
                "table_data": table_data,
                "table_markdown": dataframe_to_markdown(filtered_df),
                "table_html": dataframe_to_html(filtered_df),
                "columns": list(filtered_df.columns),
                "total_rows": len(filtered_df),
            }

        answer_col = parsed.get("answer_column")
        if answer_col and answer_col in filtered_df.columns:
            values = filtered_df[answer_col].dropna().unique()
            if len(values) == 1:
                val = values[0]
                return {
                    "success": True,
                    "response_type": "value",
                    "answer": str(val) if not pd.isna(val) else "N/A",
                }

        # Use JSON-safe conversion
        table_data = dataframe_to_json_safe_records(filtered_df)

        return {
            "success": True,
            "response_type": "table",
            "intro_message": generate_intro_message(query, len(filtered_df)),
            "table_data": table_data,
            "table_markdown": dataframe_to_markdown(filtered_df),
            "table_html": dataframe_to_html(filtered_df),
            "columns": list(filtered_df.columns),
            "total_rows": len(filtered_df),
        }

    except Exception as e:
        logger.error(f"Direct LLM query failed: {e}")
        return {"success": False, "response_type": "text", "answer": "NO_MATCH"}
