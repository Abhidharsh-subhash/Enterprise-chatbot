import os
import pandas as pd
import traceback
from app.core.openai_client import client, CHAT_MODEL
from app.core.logger import logger

UPLOADS_DIR = os.path.abspath("./uploads")


# --- THE UNIVERSAL FILTER FUNCTION ---
def smart_filter(df, date_str=None, keyword=None):
    """
    Filters a dataframe without needing to know exact column names.

    - If date_str is provided:
        Prefer columns named like 'Created_Date', 'Created Date', 'Date', 'Order_Date', etc.,
        and filter rows where that date column matches the given date.
    - If keyword is provided:
        1) Check whether that keyword appears ANYWHERE in the full df.
        2) If it never appears and we already have a non-empty date filter,
           ignore the keyword and return the date-only rows.
        3) If it appears somewhere, filter rows (intersection with any existing date filter).
    """
    filtered_df = df.copy()

    # --- 1. HANDLE DATE FILTERING ---
    if date_str:
        try:
            # parse query date similarly to your working code
            target_date = pd.to_datetime(date_str, dayfirst=True).date()
        except Exception as e:
            logger.error(f"Smart Filter Date Parse Error: {e}")
            return pd.DataFrame()

        date_found = False

        # Prefer obvious date columns first
        preferred_names = [
            "created_date",
            "created date",
            "date",
            "order_date",
            "order date",
            "booking_date",
            "booking date",
        ]
        lower_map = {c.lower(): c for c in df.columns}

        candidate_cols = []
        for name in preferred_names:
            if name in lower_map:
                candidate_cols.append(lower_map[name])

        # Fallback: all other columns
        candidate_cols += [c for c in df.columns if c not in candidate_cols]

        for col in candidate_cols:
            s = df[col]

            # If already datetime, use directly
            if pd.api.types.is_datetime64_any_dtype(s):
                temp_series = s
            else:
                try:
                    temp_series = pd.to_datetime(s, errors="coerce", dayfirst=True)
                except Exception:
                    continue

            if temp_series.notna().sum() == 0:
                continue

            mask = temp_series.dt.date == target_date
            if mask.any():
                filtered_df = df[mask]
                date_found = True
                logger.info(
                    f"smart_filter: matched date {target_date} in column '{col}', rows={mask.sum()}"
                )
                break

        if not date_found:
            logger.info(f"smart_filter: no rows found for date {target_date}")
            return pd.DataFrame()

    # --- 2. HANDLE TEXT KEYWORD FILTERING ---
    if keyword:
        kw = str(keyword).lower().strip()

        # 2a) Check if keyword exists ANYWHERE in the full df
        try:
            global_mask = (
                df.astype(str)
                .apply(lambda x: x.str.lower().str.contains(kw, na=False))
                .any(axis=1)
            )
            global_any = bool(global_mask.any())
        except Exception as e:
            logger.error(f"Smart Filter Global Keyword Error: {e}")
            global_any = False

        if not global_any:
            # Keyword literally doesn't exist in the file.
            if date_str and len(filtered_df) > 0:
                # We DO have rows for the date → return them, ignoring keyword
                logger.info(
                    f"smart_filter: keyword '{kw}' not found anywhere; "
                    f"returning date-only rows ({len(filtered_df)})."
                )
                return filtered_df
            else:
                logger.info(
                    f"smart_filter: keyword '{kw}' not found anywhere; returning empty."
                )
                return pd.DataFrame()

        # 2b) Keyword does exist somewhere → apply it to the (possibly date-filtered) df
        try:
            before = len(filtered_df)
            local_mask = (
                filtered_df.astype(str)
                .apply(lambda x: x.str.lower().str.contains(kw, na=False))
                .any(axis=1)
            )
            filtered_df = filtered_df[local_mask]
            logger.info(
                f"smart_filter: keyword='{kw}' filtered rows from {before} to {len(filtered_df)}"
            )
        except Exception as e:
            logger.error(f"Smart Filter Keyword Error: {e}")

    return filtered_df


def execute_pandas_retrieval(file_name: str, query: str) -> str:
    file_path = os.path.join(UPLOADS_DIR, file_name)

    if not os.path.exists(file_path):
        return "Error: Source file not found."

    try:
        df = pd.read_excel(file_path, header=0)
        df.columns = [str(col).strip() for col in df.columns]

        # Show a bit more context
        sample_data = df.head(5).astype(str).to_string(index=False)
        columns = list(df.columns)
        dtypes = df.dtypes.astype(str).to_dict()

        prompt = f"""
            You are a STRICT Python code generator for pandas-style analysis.

            CONTEXT
            - A pandas DataFrame `df` is ALREADY loaded from an Excel file.
            - You MUST NOT import any modules.
            - You MUST NOT read any files.
            - You MUST NOT print anything.
            - You MUST NOT define a new DataFrame from scratch.
            - Use ONLY:
                - the existing DataFrame `df`
                - the helper: smart_filter(df, date_str=None, keyword=None) -> DataFrame

            === DATAFRAME INFO ===
            - Columns: {columns}
            - dtypes: {dtypes}

            === SAMPLE ROWS (for understanding only) ===
            {sample_data}

            === USER QUERY ===
            {query!r}

            YOUR TASK
            Generate Python code (NO comments, NO backticks) that does EXACTLY this:

            1. **Extract parameters from the user query** by setting THREE variables:
            - `date_str`: 
                * If the query clearly mentions a specific date (e.g. "18th nov 2025",
                    "18/11/2025", "18-11-2025"), set date_str to that exact string.
                * Otherwise, set: date_str = None
            - `keyword`:
                * If the query mentions some service/item/subject (e.g. "ironing service"),
                    set keyword to the main word, e.g. "ironing".
                * Otherwise, set: keyword = None
            - `question_type`:
                * If the query is asking "who", "which customer", "which person", etc.,
                    set: question_type = "who"
                * If the query asks for a count ("how many", "number of", "count of"),
                    set: question_type = "count"
                * Otherwise:
                    question_type = "other"

            2. **ALWAYS call the helper**:
                result_df = smart_filter(df, date_str=date_str, keyword=keyword)

            3. **POST-PROCESSING** based on `question_type`:
            - If question_type == "who":
                * Detect person/customer columns:
                        person_cols = [c for c in result_df.columns
                                        if any(k in c.lower() for k in ["name", "customer", "client", "guest", "user", "person"])]
                * If result_df is empty: set `result = "NO_MATCH"`.
                * Else if person_cols is not empty:
                        result = result_df[person_cols].drop_duplicates().to_string(index=False)
                * Else (no obvious person columns):
                        result = result_df.to_string(index=False)

            - ELIF question_type == "count":
                * If result_df is empty: set `result = "0"`.
                * Else: set `result = str(len(result_df))`.

            - ELSE (question_type == "other"):
                * If result_df is empty: set `result = "NO_MATCH"`.
                * Else: set `result = result_df.to_string(index=False)`.

            4. **ALWAYS** define a string variable:
                result

            5. If `result_df` is empty for any reason, you MUST set:
                result = "NO_MATCH"

            IMPORTANT RULES
            - Use ONLY the existing DataFrame `df` and the helper `smart_filter`.
            - Do NOT import pandas or any other modules.
            - Do NOT read Excel or CSV files.
            - Do NOT print anything.
            - At the end of the code, the following variables MUST exist:
                date_str, keyword, question_type, result_df, result

            EXAMPLES (for pattern ONLY, DO NOT hardcode these queries)

            Example 1:
                # Query: "who placed order for ironing service on 18th nov 2025"

                date_str = "18th nov 2025"
                keyword = "ironing"
                question_type = "who"
                result_df = smart_filter(df, date_str=date_str, keyword=keyword)
                person_cols = [c for c in result_df.columns
                            if any(k in c.lower() for k in ["name", "customer", "client", "guest", "user", "person"])]
                if len(result_df) == 0:
                    result = "NO_MATCH"
                elif person_cols:
                    result = result_df[person_cols].drop_duplicates().to_string(index=False)
                else:
                    result = result_df.to_string(index=False)

            Example 2:
                # Query: "how many orders were placed on 18th nov 2025"

                date_str = "18th nov 2025"
                keyword = None
                question_type = "count"
                result_df = smart_filter(df, date_str=date_str, keyword=keyword)
                if len(result_df) == 0:
                    result = "0"
                else:
                    result = str(len(result_df))

            Now, following the same pattern as above, write code for the actual user query shown in "USER QUERY".
            Return ONLY executable Python code (no comments, no backticks).
            """
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python assistant. Output only executable code.",
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

        local_vars = {"df": df, "pd": pd, "smart_filter": smart_filter}

        try:
            exec(code, {}, local_vars)
        except Exception as exec_err:
            logger.error(f"CODE EXECUTION ERROR:\n{code}\nERROR: {exec_err}")
            return f"Error executing code: {exec_err}"

        raw_result = local_vars.get("result", "No result variable returned.")
        return str(raw_result)

    except Exception as e:
        logger.error(f"System Error: {traceback.format_exc()}")
        return f"System Error: {str(e)}"
