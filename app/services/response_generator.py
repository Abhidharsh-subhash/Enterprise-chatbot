from openai import OpenAI
from typing import List, Dict
import pandas as pd
import json
from app.core.config import settings


class ResponseGenerator:
    """Generate natural language responses from query results"""

    SYSTEM_PROMPT = """
You are a helpful data analyst assistant.
Your job is to explain query results in clear, natural language.

STRICT DATA FIDELITY RULES:
- The tabular / JSON data you receive is the ground truth.
- You MUST treat every value in the data (numbers, strings, identifiers, dates, etc.)
  as immutable facts.
- Whenever you mention any specific value from the data, you MUST copy it
  character-for-character exactly as it appears in the data.
- Do NOT round, truncate, abbreviate, or "simplify" any individual values from the data.
- Do NOT change trailing digits, omit digits, or format numbers in scientific notation
  unless they are already in that format in the data.
- Do NOT invent values that do not appear in the data.

AGGREGATIONS:
- You MAY compute new aggregated values (such as sums, averages, percentages, counts)
  based on the data when this helps answer the question.
- When you compute a new numeric value, you may round it reasonably (e.g. to 2 decimals),
  but you MUST NOT change its order of magnitude or misrepresent it.
- Never modify the original data values themselves.

GUIDELINES:
1. Be concise but informative
2. Highlight key insights and patterns
3. If there are trends or notable patterns, mention them
4. Use bullet points or numbered lists for multiple items
5. If data shows comparisons, highlight the differences
6. Be conversational but professional

FORMATTING:
- Use **bold** for important numbers or key terms
- Use bullet points for lists
- Keep responses under 300 words unless data is complex
"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = getattr(settings, "llm_model", "gpt-4o")

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

        # Special case: query returned only an 'error' column
        if list(data.columns) == ["error"] and len(data) == 1:
            msg = str(data.iloc[0]["error"])
            return f"I couldn't answer your question from the available data: {msg}"

        # Format data for LLM
        data_str = self._format_dataframe(data)

        # JSON ground truth (limit size)
        preview_df = data.head(50)
        data_records = preview_df.to_dict(orient="records")
        data_json = json.dumps(data_records, ensure_ascii=False, indent=2)

        # Build context
        history_context = ""
        if conversation_history:
            history_context = self._build_history_context(conversation_history)

        user_prompt = f"""
{history_context}

User Question: {question}

The following JSON array contains the TRUE, EXACT query result data (up to 50 rows).
This JSON is the ground truth. When you refer to any specific value from the data,
you MUST copy it exactly as it appears here, without changing digits, characters,
format, or precision:

JSON RESULT (GROUND TRUTH):
{data_json}

For convenience, here is a pretty-printed table view of the same data:
{data_str}

{f"SQL Query Used: {sql_used}" if sql_used else ""}

Please provide a clear, natural language answer that:
1. Directly answers the user's question
2. Never alters or approximates any of the values from the data
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

    def generate_error_response(
        self, question: str, error: str, attempted_sql: str = None
    ) -> str:
        """Generate helpful error response"""

        error_prompt = f"""
The user asked: "{question}"

But the query failed with error: {error}

{f"Attempted SQL: {attempted_sql}" if attempted_sql else ""}

═══════════════════════════════════════════════════════════════
IMPORTANT DATABASE RULES TO REMEMBER:
═══════════════════════════════════════════════════════════════
1. ALL column names in the database are in lowercase (e.g., `product_name`, not `Product_Name`)
2. ALL string/text values in the database are stored in lowercase (e.g., 'electronics', not 'Electronics')
3. When suggesting corrections, always use lowercase for column names and values

Generate a helpful, friendly response that:
1. Acknowledges the question couldn't be answered
2. Suggests what the user might try instead (using lowercase column names and values)

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
            # Fallback response
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

    def _format_dataframe(self, df: pd.DataFrame, max_rows: int = 25) -> str:
        """Format DataFrame for LLM consumption"""
        if len(df) > max_rows:
            display_df = df.head(max_rows)
            footer = (
                f"\n... and {len(df) - max_rows} more rows (showing first {max_rows})"
            )
        else:
            display_df = df
            footer = ""

        # Try to format nicely
        try:
            return display_df.to_markdown(index=False) + footer
        except:
            return display_df.to_string(index=False) + footer

    def _build_history_context(self, history: List[Dict]) -> str:
        """Build conversation history context"""
        if not history:
            return ""

        parts = ["Previous conversation:"]
        recent = history[-4:]  # Last 2 exchanges

        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)


# Singleton instance
response_generator = ResponseGenerator()
