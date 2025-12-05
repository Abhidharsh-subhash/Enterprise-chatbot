# response_generator.py - Updated version
from openai import OpenAI
from typing import List, Dict
import pandas as pd
import re

from app.core.config import settings


class ResponseGenerator:
    """Generate natural language responses from query results"""

    SYSTEM_PROMPT = """You are a helpful data analyst assistant.
Your job is to explain query results in clear, natural language.

**CRITICAL RULE - MENTION ALL RECORDS:**
- You MUST mention EVERY record/row in the query results
- Do NOT skip any records
- Do NOT summarize by showing only "top 3" or "some examples"
- If there are 4 orders, describe all 4 orders
- If there are 10 customers, list all 10 customers

GUIDELINES:
1. Be concise but COMPLETE - mention ALL records
2. Use numbered lists for multiple items
3. Number formatting rules:
   - For monetary values and counts: use commas (e.g., $1,234,567 or 1,234 items)
   - For decimals: round to 2 places
   - **NEVER format phone numbers, IDs, codes with commas**
4. Highlight key insights and patterns AFTER listing all records
5. Use bullet points or numbered lists for clarity

FORMATTING:
- Use **bold** for important values like names, amounts
- Use numbered lists (1., 2., 3., etc.) for listing multiple records
- After listing all records, you can add a brief summary

EXAMPLE FORMAT for multiple records:
"Found X orders/customers/records:

1. **[Name/ID]**: [key details]
2. **[Name/ID]**: [key details]
3. **[Name/ID]**: [key details]
...

Summary: [brief insight about the data]"
"""

    # Columns that should NOT be number-formatted
    EXACT_VALUE_KEYWORDS = [
        "phone",
        "mobile",
        "tel",
        "contact",
        "cell",
        "id",
        "code",
        "number",
        "num",
        "no",
        "zip",
        "pin",
        "postal",
        "reference",
        "ref",
        "tracking",
        "account",
        "acc",
        "ssn",
        "pan",
        "aadhar",
        "aadhaar",
        "serial",
        "sku",
        "barcode",
    ]

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = getattr(settings, "llm_model", "gpt-4o-mini")

    def generate_response(
        self,
        question: str,
        data: pd.DataFrame,
        sql_used: str = None,
        conversation_history: List[Dict] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate natural language response from query results"""

        # Handle empty results
        if data is None or len(data) == 0:
            return self._generate_empty_response(question)

        total_rows = len(data)

        # Format data for LLM with column type hints
        data_str, column_hints = self._format_dataframe_with_hints(data)

        # Build context
        history_context = ""
        if conversation_history:
            history_context = self._build_history_context(conversation_history)

        # Explicit instruction about record count
        completeness_instruction = f"""
**IMPORTANT: The query returned exactly {total_rows} record(s).**
You MUST describe ALL {total_rows} record(s) in your response.
Do NOT skip any records or show only examples.
"""

        user_prompt = f"""
{history_context}

User Question: {question}

{column_hints}

{completeness_instruction}

Query Results ({total_rows} rows - DESCRIBE ALL OF THEM):
{data_str}

{f"SQL Query Used: {sql_used}" if sql_used else ""}

Please provide a response that:
1. Lists ALL {total_rows} records with their key details
2. Uses exact values for phone numbers, IDs, and codes (no comma formatting)
3. Ends with a brief summary or insight if applicable
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1500,  # Increased to allow for more complete responses
            )

            answer = response.choices[0].message.content

            # Validate that response mentions the correct count
            answer = self._validate_completeness(answer, total_rows, data)

            return answer

        except Exception as e:
            return f"I found the data but encountered an error generating the response: {str(e)}\n\nHere's the raw data:\n{data_str}"

    def _validate_completeness(
        self, answer: str, expected_count: int, data: pd.DataFrame
    ) -> str:
        """
        Validate that the response mentions all records.
        If not, append a note with missing records.
        """
        # Count numbered items in response (1., 2., 3., etc.)
        numbered_items = len(re.findall(r"^\d+\.|\n\d+\.", answer))

        # If we have significantly fewer items mentioned than expected
        if expected_count > 3 and numbered_items < expected_count - 1:
            # Append a data summary
            summary = self._generate_data_summary(data)
            answer += f"\n\n---\n**Complete Data Summary ({expected_count} records):**\n{summary}"

        return answer

    def _generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate a concise summary of all records"""
        summary_parts = []

        # Find key columns (name, id, amount columns)
        key_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(
                kw in col_lower for kw in ["name", "customer", "order", "id", "number"]
            ):
                key_cols.append(col)
            elif any(
                kw in col_lower
                for kw in ["amount", "cost", "price", "total", "discount", "promo"]
            ):
                key_cols.append(col)

        # Limit to most relevant columns
        key_cols = key_cols[:5] if key_cols else list(df.columns)[:4]

        for idx, row in df.iterrows():
            parts = []
            for col in key_cols:
                val = row.get(col, "")
                if pd.notna(val):
                    if isinstance(val, float):
                        val = f"{val:.2f}"
                    parts.append(f"{col}: {val}")
            summary_parts.append(f"{idx + 1}. " + " | ".join(parts))

        return "\n".join(summary_parts)

    def _is_exact_value_column(self, column_name: str) -> bool:
        """Check if column should preserve exact values (no formatting)"""
        col_lower = column_name.lower().replace("_", " ").replace("-", " ")
        return any(keyword in col_lower for keyword in self.EXACT_VALUE_KEYWORDS)

    def _format_dataframe_with_hints(
        self, df: pd.DataFrame, max_rows: int = 50  # Increased from 25
    ) -> tuple:
        """Format DataFrame for LLM consumption with column type hints."""
        display_df = df.copy()
        exact_columns = []

        for col in display_df.columns:
            if self._is_exact_value_column(col):
                exact_columns.append(col)
                display_df[col] = display_df[col].apply(
                    lambda x: (
                        str(int(x))
                        if pd.notna(x) and self._is_numeric(x)
                        else str(x) if pd.notna(x) else ""
                    )
                )

        column_hints = ""
        if exact_columns:
            column_hints = f"""
**These columns contain identifiers/phone numbers - display values EXACTLY as shown:**
{', '.join(exact_columns)}
"""

        # Show all rows if under max_rows
        if len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
            footer = f"\n... and {len(df) - max_rows} more rows"
        else:
            footer = ""

        try:
            formatted = display_df.to_markdown(index=False) + footer
        except:
            formatted = display_df.to_string(index=False) + footer

        return formatted, column_hints

    def _is_numeric(self, value) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def generate_error_response(
        self, question: str, error: str, attempted_sql: str = None
    ) -> str:
        """Generate helpful error response"""

        error_prompt = f"""
The user asked: "{question}"

But the query failed with error: {error}

{f"Attempted SQL: {attempted_sql}" if attempted_sql else ""}

Generate a helpful, friendly response that:
1. Acknowledges the question couldn't be answered
2. Suggests what the user might try instead

Keep it brief and helpful.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Be concise and friendly.",
                    },
                    {"role": "user", "content": error_prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )

            return response.choices[0].message.content

        except:
            return f"""I wasn't able to answer your question: "{question}"

**Issue:** {error}

**Suggestions:**
- Try rephrasing your question
- Make sure you're referring to columns that exist in your data
- Check if the data you need has been uploaded"""

    def _generate_empty_response(self, question: str) -> str:
        """Generate response for empty results"""
        return f"""No results found for your query.

I searched for "{question}" but didn't find any matching records. 

**Suggestions:**
- Check if the date format is correct
- Verify the filter criteria (e.g., "promo discount" might be stored as a specific value)
- Try a broader search first to see what data exists"""

    def _build_history_context(self, history: List[Dict]) -> str:
        """Build conversation history context"""
        if not history:
            return ""

        parts = ["Previous conversation:"]
        recent = history[-4:]

        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)


# Singleton instance
response_generator = ResponseGenerator()
