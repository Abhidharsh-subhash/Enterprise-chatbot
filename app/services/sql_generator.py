from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
import json
import re

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

COMMON PATTERNS:
- Total/Sum: SELECT column, SUM(value) FROM table GROUP BY column
- Average: SELECT column, AVG(value) FROM table GROUP BY column
- Count: SELECT column, COUNT(*) FROM table GROUP BY column
- Filter: SELECT * FROM table WHERE condition
- Top N: SELECT * FROM table ORDER BY column DESC LIMIT N
- Date range: WHERE date_column BETWEEN 'start' AND 'end'
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
                parts.append(schema["prompt_text"])
            else:
                # Build from schema info
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
