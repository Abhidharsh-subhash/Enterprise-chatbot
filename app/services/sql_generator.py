# sql_generator.py
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime, timedelta

from app.core.config import settings


class SQLGenerator:
    """Generate SQL queries from natural language using LLM"""

    SYSTEM_PROMPT = """You are an expert SQL query generator for SQLite database.
Your task is to convert natural language questions into accurate SQL queries.

CRITICAL RULES:
1. Use ONLY the exact column names from the provided schema
2. SQLite syntax - use standard SQL
3. For case-insensitive string matching: column LIKE '%value%' COLLATE NOCASE
4. Always include proper GROUP BY when using aggregations with other columns
5. Use LIMIT for large results unless user asks for all data
6. Handle NULL values with IFNULL() or COALESCE()
7. Return ONLY the SQL query - no markdown, no explanations, no code blocks

SQLITE SPECIFICS:
- String concatenation: ||
- Date functions: DATE(), TIME(), DATETIME(), strftime()
- Type conversion: CAST(column AS REAL)
- Null handling: IFNULL(column, default_value)

**CRITICAL DATE RANGE HANDLING (VERY IMPORTANT):**
When user asks for data between two dates (e.g., "from 21-10-2025 to 23-11-2025"):
- BOTH start and end dates should be FULLY INCLUSIVE
- For datetime/timestamp columns, use this pattern:
  WHERE DATE(date_column) >= 'start_date' AND DATE(date_column) <= 'end_date'
  OR
  WHERE date_column >= 'start_date' AND date_column < 'day_after_end_date'
- Always convert dates to ISO format: YYYY-MM-DD
- Example: "from 21-10-2025 to 23-11-2025" should query:
  WHERE DATE(order_date) >= '2025-10-21' AND DATE(order_date) <= '2025-11-23'
  OR
  WHERE order_date >= '2025-10-21' AND order_date < '2025-11-24'

COMMON PATTERNS:
- Total/Sum: SELECT column, SUM(value) FROM table GROUP BY column
- Average: SELECT column, AVG(value) FROM table GROUP BY column
- Count: SELECT column, COUNT(*) FROM table GROUP BY column
- Filter: SELECT * FROM table WHERE condition
- Top N: SELECT * FROM table ORDER BY column DESC LIMIT N
- Date range (INCLUSIVE): WHERE DATE(date_column) BETWEEN 'start' AND 'end'
"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = getattr(settings, "llm_model", "gpt-4o-mini")

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
        # Preprocess question to handle date ranges
        processed_question = self._preprocess_date_ranges(question)

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

        # Add explicit date handling instruction
        date_instruction = self._get_date_instruction(processed_question)

        user_prompt = f"""
{schema_context}

{sample_context}

{history_context}

{date_instruction}

User Question: {processed_question}

Generate a SQLite query to answer this question. Return ONLY the SQL query.
IMPORTANT: For date ranges, ensure BOTH start and end dates are FULLY INCLUSIVE.
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

            # Post-process SQL to fix date range issues
            sql = self._fix_date_range_in_sql(sql)

            return sql, "Query generated successfully"

        except Exception as e:
            return "", f"Error generating SQL: {str(e)}"

    def _preprocess_date_ranges(self, question: str) -> str:
        """
        Preprocess question to standardize date formats and add clarity.
        """
        # Pattern to match dates in various formats
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        date_patterns = [
            (r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", self._convert_date_format),
        ]

        processed = question
        for pattern, converter in date_patterns:
            processed = re.sub(pattern, converter, processed)

        return processed

    def _convert_date_format(self, match) -> str:
        """Convert date from DD-MM-YYYY to YYYY-MM-DD (ISO format)"""
        day, month, year = match.groups()
        try:
            # Validate it's a real date
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            # If invalid date, return original
            return match.group(0)

    def _get_date_instruction(self, question: str) -> str:
        """
        Generate explicit date handling instructions if date range detected.
        """
        # Check if question contains date range indicators
        range_indicators = [
            r"from\s+\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}",
            r"between\s+\d{4}-\d{2}-\d{2}\s+and\s+\d{4}-\d{2}-\d{2}",
            r"\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}",
        ]

        for pattern in range_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                return """
**DATE RANGE INSTRUCTION:**
This query involves a date range. Ensure the end date is FULLY INCLUSIVE.
Use one of these approaches:
1. WHERE DATE(date_column) >= 'start' AND DATE(date_column) <= 'end'
2. WHERE date_column >= 'start' AND date_column < 'day_after_end'
DO NOT use simple BETWEEN for datetime columns as it may exclude end date records.
"""

        return ""

    def _fix_date_range_in_sql(self, sql: str) -> str:
        """
        Post-process SQL to fix potential date range issues.
        Converts BETWEEN clauses to inclusive range queries.
        """
        # Pattern to find BETWEEN date clauses
        between_pattern = (
            r"(\w+)\s+BETWEEN\s+'(\d{4}-\d{2}-\d{2})'\s+AND\s+'(\d{4}-\d{2}-\d{2})'"
        )

        def replace_between(match):
            column = match.group(1)
            start_date = match.group(2)
            end_date = match.group(3)

            # Calculate day after end date for inclusive range
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                next_day = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

                # Use DATE() function for safe comparison
                return f"DATE({column}) >= '{start_date}' AND DATE({column}) <= '{end_date}'"
            except:
                return match.group(0)

        fixed_sql = re.sub(between_pattern, replace_between, sql, flags=re.IGNORECASE)

        return fixed_sql

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
7. Date range issues - ensure end dates are inclusive

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

            sql = self._clean_sql(response.choices[0].message.content)
            return self._fix_date_range_in_sql(sql)

        except Exception as e:
            return ""

    def _build_schema_context(self, schemas: List[Dict[str, Any]]) -> str:
        """Build schema context for prompt"""
        parts = ["AVAILABLE TABLES AND SCHEMAS:"]
        parts.append("=" * 50)

        for schema in schemas:
            if "prompt_text" in schema:
                parts.append(schema["prompt_text"])
            else:
                # Build from schema info
                parts.append(f"\nTable: {schema.get('table_name', 'unknown')}")
                if "columns" in schema:
                    parts.append("Columns:")
                    for col in schema["columns"]:
                        if isinstance(col, dict):
                            col_type = col.get("type", "TEXT")
                            col_name = col.get("name")
                            # Add hint for date columns
                            if "date" in col_name.lower() or "time" in col_name.lower():
                                parts.append(
                                    f"  - {col_name} ({col_type}) [DATE/DATETIME COLUMN]"
                                )
                            else:
                                parts.append(f"  - {col_name} ({col_type})")
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
