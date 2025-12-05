# sql_generator.py - Complete updated version
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
2. **Match column values EXACTLY as shown in "CATEGORICAL COLUMNS" section**
3. SQLite syntax only
4. Always include proper GROUP BY when using aggregations
5. Return ONLY the SQL query - no markdown, no explanations, no code blocks

**NUMERIC CONDITION INTERPRETATION (VERY IMPORTANT):**
When user asks if someone "used", "applied", "had", "got" something:
- "used promo discount" → WHERE promo_discount > 0
- "used coupon" → WHERE coupon_discount_amount > 0
- "had express delivery" → WHERE express_cost > 0  
- "got special discount" → WHERE special_discount > 0
- "paid tax" → WHERE tax_amount > 0

NULL and ZERO are different:
- NULL means no value recorded
- 0 means no discount was applied
- To find records that USED something: column > 0 (not just IS NOT NULL)

**DATE COLUMN SELECTION:**
When user asks about when something happened, choose the right date column:
- "placed on", "created on", "ordered on", "opted for on" → order_created_date
- "closed on", "completed on", "delivered on" → order_closed_date
- "paid on", "payment on" → payment_date

**DATE HANDLING:**
- Convert dates to YYYY-MM-DD format
- For single date: DATE(column) = 'YYYY-MM-DD'
- For date range: DATE(column) >= 'start' AND DATE(column) <= 'end'
- Both start and end dates are INCLUSIVE

**VALUE MATCHING:**
- Use EXACT values from CATEGORICAL COLUMNS list
- For partial matches: LIKE '%term%' COLLATE NOCASE

SQLITE SPECIFICS:
- IFNULL(column, 0) for null handling
- DATE() function for date extraction
- COLLATE NOCASE for case-insensitive comparison
"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = getattr(settings, "llm_model", "gpt-4o-mini")

    def generate_sql(
        self,
        question: str,
        schemas: List[Dict[str, Any]],
        sample_data: Dict[str, str] = None,
        column_context: Dict[str, Any] = None,
        conversation_history: List[Dict] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, str]:
        """Generate SQL from natural language question."""

        # Preprocess question
        processed_question = self._preprocess_date_ranges(question)

        # Analyze question intent
        intent_hints = self._analyze_question_intent(processed_question)

        # Build contexts
        schema_context = self._build_schema_context(schemas)

        column_context_str = ""
        if column_context:
            column_context_str = self._format_column_context(column_context)

        sample_context = ""
        if sample_data:
            sample_context = self._build_sample_context(sample_data)

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

**REMINDERS:**
1. For "used promo discount" → use WHERE promo_discount > 0 (not IS NOT NULL)
2. For date filtering on "placed/ordered/created" → use order_created_date
3. Select ALL relevant columns so the user gets complete information

Generate a SQLite query. Return ONLY the SQL query.
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
            sql = self._fix_numeric_conditions(sql)

            return sql, "Query generated successfully"

        except Exception as e:
            return "", f"Error generating SQL: {str(e)}"

    def _analyze_question_intent(self, question: str) -> str:
        """Analyze question to provide intent hints"""
        hints = []
        question_lower = question.lower()

        # Date intent
        date_created_words = [
            "placed",
            "created",
            "ordered",
            "opted",
            "booked",
            "requested",
        ]
        date_closed_words = ["closed", "completed", "delivered", "finished"]
        date_paid_words = ["paid", "payment"]

        for word in date_created_words:
            if word in question_lower:
                hints.append(
                    f"**DATE HINT**: '{word}' refers to ORDER CREATION → use order_created_date"
                )
                break

        for word in date_closed_words:
            if word in question_lower:
                hints.append(
                    f"**DATE HINT**: '{word}' refers to ORDER COMPLETION → use order_closed_date"
                )
                break

        for word in date_paid_words:
            if word in question_lower:
                hints.append(
                    f"**DATE HINT**: '{word}' refers to PAYMENT → use payment_date"
                )
                break

        # Numeric usage intent
        usage_patterns = [
            ("promo", "promo_discount", "promo_discount > 0"),
            ("coupon", "coupon_discount", "coupon_discount_amount > 0"),
            ("express", "express_cost", "express_cost > 0"),
            ("special discount", "special_discount", "special_discount > 0"),
        ]

        for keyword, column, condition in usage_patterns:
            if keyword in question_lower and (
                "used" in question_lower
                or "applied" in question_lower
                or "had" in question_lower
                or "got" in question_lower
            ):
                hints.append(
                    f"**NUMERIC HINT**: User asking about using '{keyword}' → use WHERE {condition}"
                )

        if hints:
            return "\n**INTENT ANALYSIS:**\n" + "\n".join(hints)

        return ""

    def _fix_numeric_conditions(self, sql: str) -> str:
        """Fix common numeric condition issues"""
        # Replace IS NOT NULL with > 0 for discount/cost columns
        patterns = [
            (r"promo_discount\s+IS\s+NOT\s+NULL", "IFNULL(promo_discount, 0) > 0"),
            (
                r"coupon_discount_amount\s+IS\s+NOT\s+NULL",
                "IFNULL(coupon_discount_amount, 0) > 0",
            ),
            (r"express_cost\s+IS\s+NOT\s+NULL", "IFNULL(express_cost, 0) > 0"),
            (r"special_discount\s+IS\s+NOT\s+NULL", "IFNULL(special_discount, 0) > 0"),
            # Also handle != 0 or <> 0 patterns
            (r"promo_discount\s*!=\s*0", "IFNULL(promo_discount, 0) > 0"),
            (r"promo_discount\s*<>\s*0", "IFNULL(promo_discount, 0) > 0"),
        ]

        fixed_sql = sql
        for pattern, replacement in patterns:
            fixed_sql = re.sub(pattern, replacement, fixed_sql, flags=re.IGNORECASE)

        return fixed_sql

    def _format_column_context(self, context: Dict[str, Any]) -> str:
        """Format column context for the prompt"""
        parts = ["**COLUMN CONTEXT:**"]
        parts.append("=" * 50)

        if context.get("date_columns"):
            parts.append("\n**DATE COLUMNS (choose based on user's intent):**")
            for col_name, info in context["date_columns"].items():
                parts.append(f"  • {col_name}: {info['semantic']}")

        if context.get("categorical_columns"):
            parts.append("\n**CATEGORICAL COLUMNS (use EXACT values):**")
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

        # Pattern for DD-MM-YYYY, DD/MM/YYYY
        def convert_date(match):
            day, month, year = match.groups()
            try:
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                return match.group(0)

        # Also handle "22 nov 2025" format
        month_names = {
            "jan": "01",
            "feb": "02",
            "mar": "03",
            "apr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "aug": "08",
            "sep": "09",
            "oct": "10",
            "nov": "11",
            "dec": "12",
        }

        processed = question

        # DD-MM-YYYY or DD/MM/YYYY
        processed = re.sub(
            r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", convert_date, processed
        )

        # DD Month YYYY (e.g., 22 nov 2025)
        for month_name, month_num in month_names.items():
            pattern = rf"(\d{{1,2}})\s*{month_name}[a-z]*\s*(\d{{4}})"

            def replace_month(match, mn=month_num):
                day = match.group(1).zfill(2)
                year = match.group(2)
                return f"{year}-{mn}-{day}"

            processed = re.sub(pattern, replace_month, processed, flags=re.IGNORECASE)

        return processed

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
        column_context: str = "",
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

Please fix the SQL query. Return ONLY the corrected SQL query.
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
            sql = self._fix_date_range_in_sql(sql)
            sql = self._fix_numeric_conditions(sql)
            return sql

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
                            parts.append(
                                f"  - {col.get('name')} ({col.get('type', 'TEXT')})"
                            )
                        else:
                            parts.append(f"  - {col}")
            parts.append("")

        return "\n".join(parts)

    def _build_sample_context(self, sample_data: Dict[str, str]) -> str:
        """Build sample data context"""
        parts = ["**SAMPLE DATA:**"]
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
