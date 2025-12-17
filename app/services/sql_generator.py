from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
import json
import re

from app.core.config import settings


class SQLGenerator:
    """Generate SQL queries from natural language using LLM"""

    SYSTEM_PROMPT = """You are an expert SQL query generator for a SQLite database.
Your task is to convert natural language questions into accurate SQL queries.

You will be given:
- A description of one or more tables (their names, columns, types, and example values).
- Optional small samples of the data.

CRITICAL RULES:
1. Use ONLY the exact table and column names from the provided schema.
2. The database is SQLite: use standard SQL compatible with SQLite.
3. DO NOT modify data: only SELECT queries (and optional CTEs with WITH) are allowed.
4. For case-insensitive string matching, use: column LIKE '%value%' COLLATE NOCASE.
5. When using aggregate functions (SUM, AVG, COUNT, etc.) with other columns, include a proper GROUP BY.
6. Use LIMIT for large results unless the user explicitly asks for all data.
7. Handle NULLs with IFNULL() or COALESCE() when appropriate.
8. If multiple tables seem relevant, you may JOIN them on matching key columns (e.g., id / customer_id),
   but only if such joins are reasonable based on table and column names.
9. Never invent tables or columns that are not listed in the schema.

DATE AND DATETIME HANDLING (VERY IMPORTANT):
- Users often specify date ranges like "from 2025-11-19 to 2025-11-25".
- You MUST treat such ranges as inclusive of BOTH the start and the end dates.
- If the column is a DATE (YYYY-MM-DD only), you can use:
    WHERE date_column BETWEEN '2025-11-19' AND '2025-11-25'
  or:
    WHERE date_column >= '2025-11-19' AND date_column <= '2025-11-25'
- If the column is a DATETIME (e.g. '2025-11-25 13:45:00'), you MUST NOT compare
  the raw datetime string directly to a plain date string with <= '2025-11-25', because
  this can exclude rows on the end date.
- For DATETIME columns, either:
  * Compare using the DATE() function:
      WHERE DATE(datetime_column) BETWEEN '2025-11-19' AND '2025-11-25'
    or
  * Use an exclusive upper bound with the next day:
      WHERE datetime_column >= '2025-11-19'
        AND datetime_column < DATE('2025-11-25', '+1 day')

IF QUESTION CANNOT BE ANSWERED:
- If the user's question truly cannot be answered from the available tables/columns,
  return a valid SELECT query that returns a single row with a single column named 'error'
  containing a brief explanation, for example:
  SELECT 'Cannot answer because there is no date column' AS error;

SQLITE NOTES:
- String concatenation: col1 || col2
- Date functions: DATE(), TIME(), DATETIME(), strftime()
- Type conversion: CAST(column AS REAL)
- Null handling: IFNULL(column, default_value)

COMMON PATTERNS:
- Total/Sum:   SELECT column, SUM(value) AS total_value FROM table GROUP BY column;
- Average:     SELECT column, AVG(value) AS avg_value FROM table GROUP BY column;
- Count:       SELECT column, COUNT(*) AS count_rows FROM table GROUP BY column;
- Filter:      SELECT * FROM table WHERE condition;
- Top N:       SELECT * FROM table ORDER BY column DESC LIMIT N;
- Date range (inclusive of start AND end dates) for date-only column:
  SELECT * FROM table
  WHERE date_column BETWEEN '2025-11-19' AND '2025-11-25';
"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = getattr(settings, "llm_model", "gpt-4o")

    def generate_sql(
        self,
        question: str,
        schemas: List[Dict[str, Any]],
        sample_data: Dict[str, str] = None,
        conversation_history: List[Dict] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, str]:
        """
        Generate SQL from natural language question.

        Returns:
            Tuple of (sql_query, explanation)
        """
        # Build schema context
        schema_context = self._build_schema_context(schemas)

        # Build sample data context
        sample_context = ""
        if sample_data:
            sample_context = self._build_sample_context(sample_data)

        # Build conversation context for follow-ups
        history_context = ""
        if conversation_history:
            history_context = self._build_history_context(conversation_history)

        user_prompt = f"""
{schema_context}

{sample_context}

{history_context}

User Question: {question}

**IMPORTANT SQL GENERATION RULES:**
1. Column names MUST be in lowercase (e.g., use `product_name` not `Product_Name`)
2. All string values in WHERE, LIKE, IN clauses MUST be in lowercase (e.g., use 'electronics' not 'Electronics')
3. Use LOWER() function is NOT needed - data is already stored in lowercase
4. For text comparisons, always use lowercase values

Examples:
- Correct: SELECT * FROM table WHERE category = 'electronics'
- Wrong:   SELECT * FROM table WHERE category = 'Electronics'
- Correct: SELECT * FROM table WHERE product_name LIKE '%iphone%'
- Wrong:   SELECT * FROM table WHERE product_name LIKE '%iPhone%'

Generate a SQLite query to answer this question. Return ONLY the SQL query.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1000,
            )

            sql = self._clean_sql(response.choices[0].message.content)
            return sql, "Query generated successfully"

        except Exception as e:
            return "", f"Error generating SQL: {str(e)}"

    def fix_sql_error(
        self,
        original_sql: str,
        error_message: str,
        schema_context: str,
        temperature: float = 0.1,
    ) -> str:
        """Attempt to fix SQL query based on error message"""

        fix_prompt = f"""
The following SQL query produced an error:

ORIGINAL SQL:
{original_sql}

ERROR MESSAGE:
{error_message}

SCHEMA:
{schema_context}

Please fix the SQL query. Common issues to check:
1. Column name spelling (must match schema exactly)
2. Table name spelling
3. Missing quotes around string values
4. Missing GROUP BY clause
5. Invalid function names
6. Incorrect data types in comparisons

Return ONLY the corrected SQL query, nothing else.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": fix_prompt},
                ],
                temperature=temperature,
                max_tokens=1000,
            )

            return self._clean_sql(response.choices[0].message.content)

        except Exception as e:
            return ""

    def _build_schema_context(self, schemas: List[Dict[str, Any]]) -> str:
        """Build schema context for prompt"""
        parts = ["AVAILABLE TABLES AND SCHEMAS:"]
        parts.append("=" * 50)

        for schema in schemas:
            if "prompt_text" in schema:
                # This now comes from sqlite_store.get_table_schema_for_prompt,
                # which is already designed to be LLM-friendly and generic.
                parts.append(schema["prompt_text"])
            else:
                # Fallback if prompt_text isn't available
                parts.append(f"\nTable: {schema.get('table_name', 'unknown')}")
                if "columns" in schema:
                    parts.append("Columns:")
                    for col in schema["columns"]:
                        if isinstance(col, dict):
                            parts.append(
                                f"  - {col.get('name')} ({col.get('type', 'TEXT')})"
                            )
                        else:
                            parts.append(f"  - {col}")
            parts.append("")

        return "\n".join(parts)

    def _build_sample_context(self, sample_data: Dict[str, str]) -> str:
        """Build sample data context"""
        parts = ["SAMPLE DATA PREVIEW:"]
        parts.append("-" * 30)

        for table_name, data in sample_data.items():
            parts.append(f"\n{table_name}:")
            parts.append(data)

        return "\n".join(parts)

    def _build_history_context(self, history: List[Dict]) -> str:
        """Build conversation history context for follow-up questions"""
        if not history:
            return ""

        parts = ["PREVIOUS CONVERSATION (for context):"]
        parts.append("-" * 30)

        # Only include last few exchanges
        recent = history[-6:]  # Last 3 exchanges

        for msg in recent:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:500]  # Limit length
            parts.append(f"{role}: {content}")

        parts.append("-" * 30)
        return "\n".join(parts)

    def _clean_sql(self, response: str) -> str:
        """Clean SQL from LLM response"""
        # Remove markdown code blocks
        response = re.sub(r"```sql\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"```\s*", "", response)

        # Remove any leading/trailing whitespace
        sql = response.strip()

        # Remove trailing semicolon (we'll add if needed)
        sql = sql.rstrip(";")

        return sql


# Singleton instance
sql_generator = SQLGenerator()
