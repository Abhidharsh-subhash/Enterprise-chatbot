# sql_generator.py - Updated version
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
2. **CRITICAL: Match column values EXACTLY as shown in the "CATEGORICAL COLUMNS" section**
3. Use LIKE with wildcards for partial matching ONLY if exact match not found
4. For case-insensitive matching: LOWER(column) = LOWER('value') or COLLATE NOCASE
5. Always include proper GROUP BY when using aggregations
6. Use LIMIT for large results unless user asks for all data
7. Return ONLY the SQL query - no markdown, no explanations

**DATE COLUMN SELECTION (VERY IMPORTANT):**
When user asks about "when something happened", use the semantic meaning:
- "opted for", "ordered", "placed", "created", "initiated" → use order_created_date or created_at
- "completed", "closed", "finished", "delivered" → use order_closed_date or completed_at
- "paid", "payment made" → use payment_date

**VALUE MATCHING RULES:**
1. If user says "premium ironing" and values show "Premium Ironing Service", use the EXACT value from the list
2. If user says partial term, find the closest match from available values
3. Use LIKE '%term%' only as last resort when no close match exists

**DATE RANGE HANDLING:**
- BOTH start and end dates should be FULLY INCLUSIVE
- For datetime columns: WHERE DATE(date_column) >= 'start' AND DATE(date_column) <= 'end'
- Convert dates to ISO format: YYYY-MM-DD

SQLITE SPECIFICS:
- String concatenation: ||
- Date functions: DATE(), strftime()
- Type conversion: CAST(column AS REAL)
- Null handling: IFNULL(column, default_value)
"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = getattr(settings, "llm_model", "gpt-4o-mini")

    def generate_sql(
        self,
        question: str,
        schemas: List[Dict[str, Any]],
        sample_data: Dict[str, str] = None,
        column_context: Dict[str, Any] = None,  # NEW PARAMETER
        conversation_history: List[Dict] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, str]:
        """
        Generate SQL from natural language question.

        Args:
            question: User's natural language question
            schemas: Table schemas
            sample_data: Sample rows from tables
            column_context: Rich context about columns (distinct values, semantics)
            conversation_history: Previous conversation for context
            temperature: LLM temperature

        Returns:
            Tuple of (sql_query, explanation)
        """
        # Preprocess question to handle date formats
        processed_question = self._preprocess_date_ranges(question)

        # Analyze question intent
        intent_hints = self._analyze_question_intent(processed_question)

        # Build schema context
        schema_context = self._build_schema_context(schemas)

        # Build column context (NEW)
        column_context_str = ""
        if column_context:
            column_context_str = self._format_column_context(column_context)

        # Build sample data context
        sample_context = ""
        if sample_data:
            sample_context = self._build_sample_context(sample_data)

        # Build conversation context
        history_context = ""
        if conversation_history:
            history_context = self._build_history_context(conversation_history)

        user_prompt = f"""
{schema_context}

{column_context_str}

{sample_context}

{history_context}

{intent_hints}

User Question: {processed_question}

**IMPORTANT INSTRUCTIONS:**
1. For date filtering, use the date column that matches the USER'S INTENT (see DATE COLUMN SELECTION rules)
2. For value filtering, use EXACT values from the CATEGORICAL COLUMNS list above
3. If user mentions "premium ironing", find the exact matching value from the service column values

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
            sql = self._fix_date_range_in_sql(sql)

            return sql, "Query generated successfully"

        except Exception as e:
            return "", f"Error generating SQL: {str(e)}"

    def _analyze_question_intent(self, question: str) -> str:
        """Analyze question to provide intent hints"""
        hints = []
        question_lower = question.lower()

        # Date intent analysis
        date_created_indicators = [
            "opted for",
            "ordered",
            "placed",
            "created",
            "initiated",
            "signed up",
            "registered",
            "subscribed",
            "booked",
            "requested",
        ]
        date_closed_indicators = [
            "completed",
            "closed",
            "finished",
            "delivered",
            "fulfilled",
            "done",
            "ended",
            "resolved",
        ]
        date_paid_indicators = ["paid", "payment", "settled", "transaction"]

        for indicator in date_created_indicators:
            if indicator in question_lower:
                hints.append(
                    f"**DATE HINT**: User asked about '{indicator}' which typically refers to ORDER CREATION date (order_created_date, created_at, etc.)"
                )
                break

        for indicator in date_closed_indicators:
            if indicator in question_lower:
                hints.append(
                    f"**DATE HINT**: User asked about '{indicator}' which typically refers to ORDER COMPLETION date (order_closed_date, completed_at, etc.)"
                )
                break

        for indicator in date_paid_indicators:
            if indicator in question_lower:
                hints.append(
                    f"**DATE HINT**: User asked about '{indicator}' which typically refers to PAYMENT date (payment_date, paid_at, etc.)"
                )
                break

        # Service/product intent
        if "premium" in question_lower:
            hints.append(
                "**VALUE HINT**: User mentioned 'premium' - look for exact values containing 'Premium' in the categorical columns"
            )

        if "service" in question_lower:
            hints.append(
                "**VALUE HINT**: User mentioned 'service' - filter on the service/service_type column using exact values from the list"
            )

        if hints:
            return "\n**INTENT ANALYSIS:**\n" + "\n".join(hints)

        return ""

    def _format_column_context(self, context: Dict[str, Any]) -> str:
        """Format column context for the prompt"""
        parts = ["**COLUMN CONTEXT (USE THIS FOR ACCURATE QUERIES):**"]
        parts.append("=" * 50)

        # Date columns with semantic meaning
        if context.get("date_columns"):
            parts.append("\n**DATE COLUMNS:**")
            for col_name, info in context["date_columns"].items():
                parts.append(f"  • {col_name}: {info['semantic']}")

        # Categorical columns with exact values
        if context.get("categorical_columns"):
            parts.append(
                "\n**CATEGORICAL COLUMNS (use these EXACT values in queries):**"
            )
            for col_name, info in context["categorical_columns"].items():
                values = info["values"]
                if len(values) <= 10:
                    values_str = ", ".join([f"'{v}'" for v in values])
                else:
                    values_str = ", ".join([f"'{v}'" for v in values[:10]])
                    values_str += f" ... (+{len(values) - 10} more)"
                parts.append(f"  • {col_name}: [{values_str}]")

        parts.append("=" * 50)
        return "\n".join(parts)

    def _preprocess_date_ranges(self, question: str) -> str:
        """Preprocess question to standardize date formats"""
        date_patterns = [
            (r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", self._convert_date_format),
        ]

        processed = question
        for pattern, converter in date_patterns:
            processed = re.sub(pattern, converter, processed)

        return processed

    def _convert_date_format(self, match) -> str:
        """Convert date from DD-MM-YYYY to YYYY-MM-DD"""
        day, month, year = match.groups()
        try:
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            return match.group(0)

    def _fix_date_range_in_sql(self, sql: str) -> str:
        """Post-process SQL to fix date range issues"""
        between_pattern = (
            r"(\w+)\s+BETWEEN\s+'(\d{4}-\d{2}-\d{2})'\s+AND\s+'(\d{4}-\d{2}-\d{2})'"
        )

        def replace_between(match):
            column = match.group(1)
            start_date = match.group(2)
            end_date = match.group(3)
            return (
                f"DATE({column}) >= '{start_date}' AND DATE({column}) <= '{end_date}'"
            )

        return re.sub(between_pattern, replace_between, sql, flags=re.IGNORECASE)

    def fix_sql_error(
        self,
        original_sql: str,
        error_message: str,
        schema_context: str,
        column_context: str = "",  # NEW
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

{column_context}

Please fix the SQL query. Common issues:
1. Column name spelling (must match schema exactly)
2. Value spelling (must match exact values from categorical columns)
3. Wrong date column for the user's intent
4. Missing quotes around string values
5. Missing GROUP BY clause

Return ONLY the corrected SQL query.
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

        except Exception:
            return ""

    def _build_schema_context(self, schemas: List[Dict[str, Any]]) -> str:
        """Build schema context for prompt"""
        parts = ["**AVAILABLE TABLES AND SCHEMAS:**"]
        parts.append("=" * 50)

        for schema in schemas:
            if "prompt_text" in schema:
                parts.append(schema["prompt_text"])
            else:
                parts.append(f"\nTable: {schema.get('table_name', 'unknown')}")
                if "columns" in schema:
                    parts.append("Columns:")
                    for col in schema["columns"]:
                        if isinstance(col, dict):
                            col_type = col.get("type", "TEXT")
                            col_name = col.get("name")
                            parts.append(f"  - {col_name} ({col_type})")
                        else:
                            parts.append(f"  - {col}")
            parts.append("")

        return "\n".join(parts)

    def _build_sample_context(self, sample_data: Dict[str, str]) -> str:
        """Build sample data context"""
        parts = ["**SAMPLE DATA PREVIEW:**"]
        parts.append("-" * 30)

        for table_name, data in sample_data.items():
            parts.append(f"\n{table_name}:")
            parts.append(data)

        return "\n".join(parts)

    def _build_history_context(self, history: List[Dict]) -> str:
        """Build conversation history context"""
        if not history:
            return ""

        parts = ["**PREVIOUS CONVERSATION:**"]
        parts.append("-" * 30)

        recent = history[-6:]
        for msg in recent:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:500]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _clean_sql(self, response: str) -> str:
        """Clean SQL from LLM response"""
        response = re.sub(r"```sql\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"```\s*", "", response)
        sql = response.strip()
        sql = sql.rstrip(";")
        return sql


# Singleton instance
sql_generator = SQLGenerator()
