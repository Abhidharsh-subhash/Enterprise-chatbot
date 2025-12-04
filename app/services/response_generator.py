# response_generator.py
from openai import OpenAI
from typing import List, Dict
import pandas as pd
import re

from app.core.config import settings


class ResponseGenerator:
    """Generate natural language responses from query results"""

    SYSTEM_PROMPT = """You are a helpful data analyst assistant.
Your job is to explain query results in clear, natural language.

GUIDELINES:
1. Be concise but informative
2. Highlight key insights and patterns
3. Number formatting rules:
   - For monetary values and counts: use commas (e.g., $1,234,567 or 1,234 items)
   - For decimals: round to 2 places
   - **CRITICAL: NEVER format phone numbers, IDs, codes, or reference numbers with commas**
   - Phone numbers, mobile numbers, customer IDs, order IDs, zip codes, PIN codes must be displayed EXACTLY as shown in the data
4. If there are trends or notable patterns, mention them
5. Use bullet points or numbered lists for multiple items
6. If data shows comparisons, highlight the differences
7. Be conversational but professional

IMPORTANT EXCEPTIONS - Do NOT add commas or modify these:
- Phone/mobile numbers (display as-is, e.g., 9025538325)
- Customer/Order/Product IDs
- Zip codes, PIN codes
- Reference numbers, tracking numbers
- Any column marked with [EXACT] prefix

FORMATTING:
- Use **bold** for important values
- Use bullet points for lists
- Keep responses under 300 words unless data is complex
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

        # Format data for LLM with column type hints
        data_str, column_hints = self._format_dataframe_with_hints(data)

        # Build context
        history_context = ""
        if conversation_history:
            history_context = self._build_history_context(conversation_history)

        user_prompt = f"""
{history_context}

User Question: {question}

{column_hints}

Query Results ({len(data)} rows):
{data_str}

{f"SQL Query Used: {sql_used}" if sql_used else ""}

Please provide a clear, natural language answer that:
1. Directly answers the user's question
2. Uses exact values for phone numbers, IDs, and codes (no comma formatting)
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=800,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"I found the data but encountered an error generating the response: {str(e)}\n\nHere's the raw data:\n{data_str}"

    def _is_exact_value_column(self, column_name: str) -> bool:
        """Check if column should preserve exact values (no formatting)"""
        col_lower = column_name.lower().replace("_", " ").replace("-", " ")
        return any(keyword in col_lower for keyword in self.EXACT_VALUE_KEYWORDS)

    def _format_dataframe_with_hints(
        self, df: pd.DataFrame, max_rows: int = 25
    ) -> tuple:
        """
        Format DataFrame for LLM consumption with column type hints.
        Returns (formatted_data, column_hints)
        """
        # Create a copy to avoid modifying original
        display_df = df.copy()

        exact_columns = []

        # Process each column
        for col in display_df.columns:
            if self._is_exact_value_column(col):
                exact_columns.append(col)
                # Convert to string to preserve exact values
                display_df[col] = display_df[col].apply(
                    lambda x: (
                        f"[EXACT]{int(x)}[/EXACT]"
                        if pd.notna(x) and self._is_numeric(x)
                        else str(x) if pd.notna(x) else ""
                    )
                )

        # Build column hints
        column_hints = ""
        if exact_columns:
            column_hints = f"""
**IMPORTANT - These columns contain identifiers/phone numbers - display values EXACTLY as shown:**
{', '.join(exact_columns)}
"""

        # Limit rows
        if len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
            footer = (
                f"\n... and {len(df) - max_rows} more rows (showing first {max_rows})"
            )
        else:
            footer = ""

        # Format output
        try:
            formatted = display_df.to_markdown(index=False) + footer
        except:
            formatted = display_df.to_string(index=False) + footer

        # Clean up markers for readability but keep the hint
        formatted = formatted.replace("[EXACT]", "").replace("[/EXACT]", "")

        return formatted, column_hints

    def _is_numeric(self, value) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _format_dataframe(self, df: pd.DataFrame, max_rows: int = 25) -> str:
        """Format DataFrame for LLM consumption (legacy method)"""
        formatted, _ = self._format_dataframe_with_hints(df, max_rows)
        return formatted

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
I searched for "{question}" but didn't find any matching records. Would you like to rephrase your question to explore available data?"""

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
