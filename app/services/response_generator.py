from openai import OpenAI
from typing import List, Dict
import pandas as pd

from app.core.config import settings


class ResponseGenerator:
    """Generate natural language responses from query results"""

    SYSTEM_PROMPT = """You are a helpful data analyst assistant.
Your job is to explain query results in clear, natural language.

GUIDELINES:
1. Be concise but informative
2. Highlight key insights and patterns
3. Format numbers appropriately (use commas for thousands, round decimals to 2 places)
4. If there are trends or notable patterns, mention them
5. Use bullet points or numbered lists for multiple items
6. If data shows comparisons, highlight the differences
7. Be conversational but professional

FORMATTING:
- Use **bold** for important numbers or key terms
- Use bullet points for lists
- Keep responses under 300 words unless data is complex
"""

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

        # Format data for LLM
        data_str = self._format_dataframe(data)

        # Build context
        history_context = ""
        if conversation_history:
            history_context = self._build_history_context(conversation_history)

        user_prompt = f"""
{history_context}

User Question: {question}

Query Results ({len(data)} rows):
{data_str}

{f"SQL Query Used: {sql_used}" if sql_used else ""}

Please provide a clear, natural language answer that:
1. Directly answers the user's question
2. Highlights key findings from the data
3. Mentions any notable patterns or insights
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

Generate a helpful, friendly response that:
1. Acknowledges the question couldn't be answered
2. Explains the issue in simple terms (not technical jargon)
3. Suggests what the user might try instead

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
        return f"""I searched the data but found no results matching your question: "{question}"

This could mean:
- There's no data matching your criteria
- The filter conditions might be too restrictive
- You might want to try a broader search

Would you like to try a different question or see what data is available?"""

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
