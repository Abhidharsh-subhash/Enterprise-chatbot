# pandas_executor.py
import os
import re
import pandas as pd
import traceback
from typing import Optional, List, Dict, Any, Tuple
from app.core.openai_client import client, CHAT_MODEL
from app.core.logger import logger

UPLOADS_DIR = os.path.abspath("./uploads")

# Your fixed Excel date format
EXCEL_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


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
                "null_count": series.isna().sum(),
                "total_count": len(series),
                "unique_count": series.nunique(),
                "unique_ratio": series.nunique() / max(len(series), 1),
            }

            # Detect if numeric
            if pd.api.types.is_numeric_dtype(series):
                profile["type"] = "numeric"
                profile["min"] = float(series.min()) if len(non_null) > 0 else None
                profile["max"] = float(series.max()) if len(non_null) > 0 else None
                profile["mean"] = float(series.mean()) if len(non_null) > 0 else None

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
                        "selectivity": contains_count
                        / len(self.df),  # Lower = more selective
                        "matched_values": [str(v) for v in matched_values],
                        "mask": contains_mask,
                    }
                )

        # Sort by selectivity (most selective first) - prefer columns where the term is rare
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
                            "selectivity": mask.sum() / len(self.df),
                            "mask": mask,
                        }
                    )
            except Exception:
                continue

        # Sort by selectivity
        findings.sort(key=lambda x: x["selectivity"])

        return findings

    def find_date_in_dataframe(self, target_date) -> List[Dict]:
        """
        Search for a date across ALL date columns.
        target_date should be a datetime.date object.
        """
        findings = []

        for col in self.columns:
            profile = self.column_profiles[col]

            if profile["type"] != "datetime":
                continue

            try:
                # Parse the column
                parsed = pd.to_datetime(
                    self.df[col], format=EXCEL_DATE_FORMAT, errors="coerce"
                )

                # Compare dates
                mask = parsed.dt.date == target_date

                if mask.sum() > 0:
                    findings.append(
                        {
                            "column": col,
                            "match_count": int(mask.sum()),
                            "selectivity": mask.sum() / len(self.df),
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
        All information comes from actual data analysis.
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


def extract_query_components_with_llm(
    query: str, engine: DynamicQueryEngine
) -> Dict[str, Any]:
    """
    Use LLM to understand the query in context of the actual data.
    The LLM sees the real data structure and extracts components accordingly.
    """
    data_summary = engine.get_data_summary_for_llm()

    # Show sample rows
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
   - Handle formats like "14th nov 2025", "november 14, 2025", etc.

3. "numeric_values": List of numbers mentioned (excluding years), or empty list
   - Include amounts, quantities, prices, etc.

4. "target_attribute": The specific piece of information the user wants to know
   - This should be a description like "payment method", "status", "total amount", etc.
   - Or null if they want all information

5. "question_type": One of "specific_value", "count", "sum", "list", "filter_only"
   - "specific_value": User wants one specific attribute (e.g., "what is the payment mode")
   - "count": User wants a count (e.g., "how many orders")
   - "sum": User wants a total (e.g., "total amount")
   - "list": User wants to see matching records
   - "filter_only": User just wants filtered data

IMPORTANT:
- Base your extraction on the ACTUAL column names and values shown above
- The search_terms should be things that would actually appear in the data
- Be precise - extract exactly what's in the question

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

        # Clean markdown if present
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
            "numeric_values": [],
            "target_attribute": None,
            "question_type": "list",
        }


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

        # Verify the column exists
        if result in engine.columns:
            return result

        # Try case-insensitive match
        for col in engine.columns:
            if col.lower() == result.lower():
                return col

        return None

    except Exception as e:
        logger.error(f"Target column identification failed: {e}")
        return None


def execute_dynamic_query(
    engine: DynamicQueryEngine, components: Dict[str, Any], query: str
) -> Dict[str, Any]:
    """
    Execute the query using extracted components.
    All filtering is data-driven based on where values are actually found.
    """
    result = {
        "success": False,
        "answer": "NO_MATCH",
        "debug": {
            "components": components,
            "filters_applied": [],
            "rows_progression": [len(engine.df)],
        },
    }

    current_df = engine.df.copy()
    combined_mask = pd.Series(True, index=engine.df.index)

    # 1. Apply text search filters
    search_terms = components.get("search_terms", [])
    for term in search_terms:
        if not term or len(str(term).strip()) < 2:
            continue

        findings = engine.find_value_in_dataframe(str(term))

        if findings:
            # Use the most selective column (first in sorted list)
            best_finding = findings[0]

            # Only apply if it's selective enough (not matching everything)
            if best_finding["selectivity"] < 0.8:
                combined_mask &= best_finding["mask"]
                result["debug"]["filters_applied"].append(
                    {
                        "type": "text_search",
                        "term": term,
                        "column": best_finding["column"],
                        "matches": best_finding["contains_matches"],
                        "selectivity": best_finding["selectivity"],
                    }
                )

                temp_count = combined_mask.sum()
                result["debug"]["rows_progression"].append(int(temp_count))

                logger.info(
                    f"Text filter '{term}' on column '{best_finding['column']}': "
                    f"{best_finding['contains_matches']} matches, {temp_count} remaining"
                )

                if temp_count == 0:
                    logger.warning(f"Filter '{term}' eliminated all rows, reverting")
                    combined_mask |= best_finding["mask"]  # Revert by OR-ing back

    # 2. Apply date filter
    date_value = components.get("date_value")
    if date_value:
        try:
            target_date = pd.to_datetime(date_value).date()
            date_findings = engine.find_date_in_dataframe(target_date)

            if date_findings:
                best_date = date_findings[0]

                # Apply date filter
                new_mask = combined_mask & best_date["mask"]

                if new_mask.sum() > 0:
                    combined_mask = new_mask
                    result["debug"]["filters_applied"].append(
                        {
                            "type": "date_filter",
                            "date": date_value,
                            "column": best_date["column"],
                            "matches": best_date["match_count"],
                        }
                    )
                    result["debug"]["rows_progression"].append(int(combined_mask.sum()))

                    logger.info(
                        f"Date filter '{date_value}' on column '{best_date['column']}': "
                        f"{combined_mask.sum()} remaining"
                    )
                else:
                    logger.warning(f"Date filter would eliminate all rows, skipping")

        except Exception as e:
            logger.error(f"Date parsing error: {e}")

    # 3. Apply numeric filter (for verification/refinement)
    numeric_values = components.get("numeric_values", [])
    if numeric_values and combined_mask.sum() > 1:  # Only if multiple rows remain
        for num in numeric_values:
            try:
                num_findings = engine.find_number_in_dataframe(float(num))

                if num_findings:
                    best_num = num_findings[0]

                    # Only apply if selective
                    if best_num["selectivity"] < 0.5:
                        new_mask = combined_mask & best_num["mask"]

                        if new_mask.sum() > 0:
                            combined_mask = new_mask
                            result["debug"]["filters_applied"].append(
                                {
                                    "type": "numeric_filter",
                                    "value": num,
                                    "column": best_num["column"],
                                    "matches": best_num["match_count"],
                                }
                            )
                            result["debug"]["rows_progression"].append(
                                int(combined_mask.sum())
                            )

                            logger.info(
                                f"Numeric filter '{num}' on column '{best_num['column']}': "
                                f"{combined_mask.sum()} remaining"
                            )

            except Exception as e:
                logger.debug(f"Numeric filter error: {e}")

    # 4. Get filtered dataframe
    filtered_df = engine.df[combined_mask]
    result["debug"]["final_row_count"] = len(filtered_df)

    if filtered_df.empty:
        result["answer"] = "NO_MATCH"
        return result

    # 5. Extract the target attribute
    target_description = components.get("target_attribute")
    question_type = components.get("question_type", "list")

    if question_type == "count":
        result["answer"] = str(len(filtered_df))
        result["success"] = True
        return result

    if question_type == "sum" and target_description:
        # Find numeric column matching the description
        target_col = find_target_column_with_llm(query, target_description, engine)
        if target_col and engine.column_profiles[target_col]["type"] == "numeric":
            total = filtered_df[target_col].sum()
            result["answer"] = str(total)
            result["success"] = True
            return result

    if target_description:
        target_col = find_target_column_with_llm(query, target_description, engine)
        result["debug"]["target_column"] = target_col

        if target_col and target_col in filtered_df.columns:
            values = filtered_df[target_col].dropna().unique()

            if len(values) == 1:
                result["answer"] = str(values[0])
                result["success"] = True
            elif len(values) > 1:
                result["answer"] = (
                    f"Multiple values found: {', '.join(str(v) for v in values[:10])}"
                )
                result["success"] = True
            else:
                result["answer"] = "No value found in the matching row(s)"

            return result

    # 6. Default: return filtered data
    if len(filtered_df) <= 10:
        result["answer"] = filtered_df.to_string(index=False)
    else:
        result["answer"] = (
            f"Found {len(filtered_df)} matching rows:\n\n{filtered_df.head(10).to_string(index=False)}"
        )

    result["success"] = True
    return result


def execute_pandas_retrieval(file_name: str, query: str) -> str:
    """
    Main entry point for dynamic pandas-based data retrieval.
    """
    file_path = os.path.join(UPLOADS_DIR, file_name)

    if not os.path.exists(file_path):
        return "Error: Source file not found."

    try:
        # Load Excel file
        df = pd.read_excel(file_path, header=0)
        df.columns = [str(col).strip() for col in df.columns]

        logger.info(f"Loaded file: {file_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Initialize dynamic query engine
        engine = DynamicQueryEngine(df)

        # Log discovered column types
        for col, profile in engine.column_profiles.items():
            logger.info(
                f"Column '{col}': type={profile['type']}, unique={profile['unique_count']}"
            )

        # Step 1: Extract query components using LLM (with data context)
        components = extract_query_components_with_llm(query, engine)

        # Step 2: Execute the query dynamically
        result = execute_dynamic_query(engine, components, query)

        logger.info(f"Query result: success={result['success']}")
        logger.info(f"Debug info: {result['debug']}")

        # Step 3: If failed, try alternative approach
        if not result["success"] or result["answer"] == "NO_MATCH":
            logger.info(
                "Primary approach failed, trying direct LLM query generation..."
            )
            alternative_result = _direct_llm_query(df, engine, query)
            if alternative_result and alternative_result != "NO_MATCH":
                return alternative_result

        return result["answer"]

    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return f"Error: {str(e)}"


def _direct_llm_query(df: pd.DataFrame, engine: DynamicQueryEngine, query: str) -> str:
    """
    Fallback: Let LLM generate specific filter conditions based on actual data.
    """
    try:
        # Show the LLM actual data samples
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
                logger.warning(f"Column '{col}' not found")
                continue

            col_profile = engine.column_profiles.get(col, {})

            # Handle date columns
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
                        logger.info(f"Date filter on '{col}': {len(filtered_df)} rows")
                except Exception as e:
                    logger.error(f"Date filter error: {e}")

            # Handle numeric columns
            elif col_profile.get("type") == "numeric":
                try:
                    num_val = float(val)
                    mask = (filtered_df[col] - num_val).abs() < 0.01
                    new_df = filtered_df[mask]
                    if len(new_df) > 0:
                        filtered_df = new_df
                        logger.info(
                            f"Numeric filter on '{col}': {len(filtered_df)} rows"
                        )
                except Exception as e:
                    logger.error(f"Numeric filter error: {e}")

            # Handle text columns
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
                    logger.info(
                        f"Text filter on '{col}' with '{val}': {len(filtered_df)} rows"
                    )

        if filtered_df.empty:
            return "NO_MATCH"

        # Extract answer
        answer_col = parsed.get("answer_column")
        if answer_col and answer_col in filtered_df.columns:
            values = filtered_df[answer_col].dropna().unique()
            if len(values) == 1:
                return str(values[0])
            elif len(values) > 1:
                return f"Multiple values: {', '.join(str(v) for v in values[:10])}"

        # Return data if no specific answer column
        if len(filtered_df) <= 10:
            return filtered_df.to_string(index=False)

        return f"Found {len(filtered_df)} rows:\n{filtered_df.head(10).to_string(index=False)}"

    except Exception as e:
        logger.error(f"Direct LLM query failed: {e}")
        return "NO_MATCH"
