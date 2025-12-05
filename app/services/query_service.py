# app/services/query_service.py
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from app.utils.embeddings import get_embedding
from app.vector_store.chrome_store import query_excel_schemas, query_user_vectors
from app.vector_store.sqlite_store import sqlite_store
from app.services.sql_generator import sql_generator
from app.services.response_generator import response_generator
from app.services.session_manager import session_manager
from app.core.config import settings


# Inline Column Analyzer to avoid import issues
class ColumnAnalyzer:
    """Analyze columns to provide context for SQL generation"""

    CATEGORICAL_KEYWORDS = [
        "service",
        "type",
        "status",
        "category",
        "method",
        "mode",
        "payment",
        "level",
        "tier",
        "plan",
        "grade",
        "priority",
    ]

    DATE_SEMANTICS = {
        "created": "when the order/record was CREATED or PLACED by customer",
        "placed": "when the order was PLACED by customer",
        "ordered": "when the order was PLACED by customer",
        "closed": "when the order was COMPLETED/CLOSED/DELIVERED",
        "completed": "when the order was COMPLETED",
        "delivered": "when the order was DELIVERED",
        "paid": "when the PAYMENT was made",
        "payment": "when the PAYMENT was made",
    }

    def get_column_context(self, table_name: str, columns: list, user_id: str) -> dict:
        """Get distinct values for categorical columns and semantics for date columns"""

        context = {
            "categorical_columns": {},
            "date_columns": {},
            "numeric_columns": [],
            "text_columns": [],
        }

        for col in columns:
            col_name = col.get("name", col) if isinstance(col, dict) else col
            col_type = col.get("type", "TEXT") if isinstance(col, dict) else "TEXT"
            col_lower = col_name.lower()

            # Identify date columns with semantic meaning
            if any(kw in col_lower for kw in ["date", "time", "_at", "_on"]):
                semantic = self._get_date_semantic(col_name)
                context["date_columns"][col_name] = {
                    "semantic": semantic,
                    "type": col_type,
                }

            # Get distinct values for categorical columns
            elif any(kw in col_lower for kw in self.CATEGORICAL_KEYWORDS):
                try:
                    query = f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT 25'
                    result_df, error = sqlite_store.execute_query(query, user_id)

                    if (
                        result_df is not None
                        and len(result_df) > 0
                        and len(result_df) <= 20
                    ):
                        values = [
                            str(v)
                            for v in result_df[col_name].tolist()
                            if v is not None
                        ]
                        if values:
                            context["categorical_columns"][col_name] = {
                                "values": values,
                                "count": len(values),
                            }
                except Exception as e:
                    print(f"Warning: Could not get distinct values for {col_name}: {e}")

            # Classify other columns
            elif col_type in ["INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"]:
                context["numeric_columns"].append(col_name)
            else:
                context["text_columns"].append(col_name)

        return context

    def _get_date_semantic(self, col_name: str) -> str:
        """Get semantic meaning of a date column"""
        col_lower = col_name.lower()

        for keyword, semantic in self.DATE_SEMANTICS.items():
            if keyword in col_lower:
                return semantic

        return "date/time field"

    def format_context_for_prompt(self, context: dict) -> str:
        """Format column context for LLM prompt"""
        parts = []

        # Date columns with semantics
        if context.get("date_columns"):
            parts.append("\n**DATE COLUMNS (choose based on user intent):**")
            for col_name, info in context["date_columns"].items():
                parts.append(f"  - {col_name}: {info['semantic']}")

        # Categorical columns with values
        if context.get("categorical_columns"):
            parts.append(
                "\n**CATEGORICAL COLUMNS (use these EXACT values in WHERE clauses):**"
            )
            for col_name, info in context["categorical_columns"].items():
                values = info["values"]
                if len(values) <= 10:
                    values_str = ", ".join([f"'{v}'" for v in values])
                else:
                    values_str = ", ".join([f"'{v}'" for v in values[:10]])
                    values_str += f" ... (+{len(values) - 10} more)"
                parts.append(f"  - {col_name}: [{values_str}]")

        return "\n".join(parts)


# Create singleton instance
column_analyzer = ColumnAnalyzer()


class QueryService:
    """Main service for handling user queries"""

    MAX_RETRIES = 3

    def __init__(self):
        pass

    async def process_question(
        self,
        question: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.1,
        show_sql: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user questions.
        Automatically detects whether to use SQL (Excel) or RAG (documents).
        """

        # Get or create session
        session_id = session_manager.get_or_create_session(session_id, user_id)

        # Get conversation history
        history = session_manager.get_history(session_id, max_messages=6)

        # Check what data types user has
        user_tables = sqlite_store.get_user_tables(user_id)
        has_excel_data = len(user_tables) > 0

        # Generate embedding for the question
        question_embedding = get_embedding(question)

        # Determine query approach
        if has_excel_data:
            # Try Excel/SQL approach first
            result = await self._process_excel_query(
                question=question,
                question_embedding=question_embedding,
                user_id=user_id,
                history=history,
                top_k=top_k,
                temperature=temperature,
                show_sql=show_sql,
            )

            if result["success"]:
                # Store in session
                session_manager.add_message(session_id, "user", question)
                session_manager.add_message(
                    session_id,
                    "assistant",
                    result["answer"],
                    metadata={"sql": result.get("sql"), "type": "excel_sql"},
                )
                result["session_id"] = session_id
                result["query_type"] = "excel_sql"
                return result

        # Fallback or primary: Try document RAG approach
        result = await self._process_document_query(
            question=question,
            question_embedding=question_embedding,
            user_id=user_id,
            history=history,
            top_k=top_k,
            temperature=temperature,
        )

        # Store in session
        session_manager.add_message(session_id, "user", question)
        session_manager.add_message(
            session_id, "assistant", result["answer"], metadata={"type": "document_rag"}
        )

        result["session_id"] = session_id
        result["query_type"] = "document_rag"
        return result

    async def _process_excel_query(
        self,
        question: str,
        question_embedding: List[float],
        user_id: str,
        history: List[Dict],
        top_k: int,
        temperature: float,
        show_sql: bool,
    ) -> Dict[str, Any]:
        """Process query using SQL on Excel data"""

        # Step 1: Find relevant tables using semantic search
        relevant_tables = query_excel_schemas(
            query_embedding=question_embedding,
            user_id=user_id,
            top_k=min(top_k, 3),
        )

        if not relevant_tables:
            return {
                "success": False,
                "answer": "No relevant Excel data found for your question.",
                "data": None,
                "sql": None,
                "error": "No matching tables",
            }

        # Step 2: Get sample data for context
        sample_data = {}
        for table in relevant_tables[:2]:
            sample_df = sqlite_store.get_sample_data(table["table_name"], 3)
            if sample_df is not None:
                sample_data[table["table_name"]] = sample_df.to_string(index=False)

        # Step 3: Get rich column context (distinct values, semantics)
        # FIXED: Proper initialization
        column_context = {
            "categorical_columns": {},
            "date_columns": {},
            "numeric_columns": [],
            "text_columns": [],
        }

        for table in relevant_tables[:2]:
            table_name = table["table_name"]
            columns = table.get("columns", [])

            try:
                ctx = column_analyzer.get_column_context(
                    table_name=table_name, columns=columns, user_id=user_id
                )

                # Merge contexts properly
                if ctx.get("categorical_columns"):
                    column_context["categorical_columns"].update(
                        ctx["categorical_columns"]
                    )

                if ctx.get("date_columns"):
                    column_context["date_columns"].update(ctx["date_columns"])

                if ctx.get("numeric_columns"):
                    column_context["numeric_columns"].extend(ctx["numeric_columns"])

                if ctx.get("text_columns"):
                    column_context["text_columns"].extend(ctx["text_columns"])

            except Exception as e:
                print(f"Warning: Could not get column context for {table_name}: {e}")

        # Step 4: Generate SQL with enhanced context
        sql, explanation = sql_generator.generate_sql(
            question=question,
            schemas=relevant_tables,
            sample_data=sample_data,
            column_context=column_context,
            conversation_history=history,
            temperature=temperature,
        )

        print(f"Generated SQL: {sql}")

        if not sql:
            return {
                "success": False,
                "answer": "I couldn't generate a query for your question. Could you try rephrasing it?",
                "data": None,
                "sql": None,
                "error": explanation,
            }

        # Step 5: Execute with retry
        result_df, final_sql, error = self._execute_with_retry(
            sql=sql,
            user_id=user_id,
            schemas=relevant_tables,
            column_context=column_context,
        )

        if error:
            error_response = response_generator.generate_error_response(
                question=question, error=error, attempted_sql=final_sql
            )
            return {
                "success": False,
                "answer": error_response,
                "data": None,
                "sql": final_sql if show_sql else None,
                "error": error,
            }

        # Step 6: Generate natural language response
        answer = response_generator.generate_response(
            question=question,
            data=result_df,
            sql_used=final_sql,
            conversation_history=history,
            temperature=0.7,
        )

        # Prepare data result
        data_result = None
        if result_df is not None and len(result_df) > 0:
            data_result = {
                "columns": list(result_df.columns),
                "rows": result_df.head(100).to_dict("records"),
                "row_count": len(result_df),
            }

        return {
            "success": True,
            "answer": answer,
            "data": data_result,
            "sql": final_sql if show_sql else None,
            "sources": [
                {
                    "table": t["table_name"],
                    "file": t["original_filename"],
                    "similarity": t["similarity"],
                }
                for t in relevant_tables
            ],
            "error": None,
        }

    def _execute_with_retry(
        self,
        sql: str,
        user_id: str,
        schemas: List[Dict],
        column_context: Dict[str, Any] = None,
    ) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
        """Execute SQL with automatic retry on failure"""

        current_sql = sql
        schema_context = "\n\n".join([s.get("prompt_text", str(s)) for s in schemas])

        # Format column context for retry
        column_context_str = ""
        if column_context:
            try:
                column_context_str = column_analyzer.format_context_for_prompt(
                    column_context
                )
            except Exception as e:
                print(f"Warning: Could not format column context: {e}")

        for attempt in range(self.MAX_RETRIES):
            # Execute query
            result_df, error = sqlite_store.execute_query(current_sql, user_id)

            if error is None:
                # Success!
                return result_df, current_sql, None

            # Try to fix on failure (except last attempt)
            if attempt < self.MAX_RETRIES - 1:
                fixed_sql = sql_generator.fix_sql_error(
                    original_sql=current_sql,
                    error_message=error,
                    schema_context=schema_context,
                    column_context=column_context_str,
                )

                if fixed_sql and fixed_sql != current_sql:
                    current_sql = fixed_sql
                    continue

            # Return error
            return None, current_sql, error

        return None, current_sql, "Max retries exceeded"

    async def _process_document_query(
        self,
        question: str,
        question_embedding: List[float],
        user_id: str,
        history: List[Dict],
        top_k: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Process query using RAG on document data"""

        # Query vector store for relevant chunks
        results = query_user_vectors(
            query_embedding=question_embedding, user_id=user_id, top_k=top_k
        )

        # Check if we got results
        if not results or not results.get("documents") or not results["documents"][0]:
            return {
                "success": False,
                "answer": "I couldn't find any relevant information in your uploaded documents. Please make sure you've uploaded the necessary files.",
                "data": None,
                "sources": None,
                "error": "No matching documents",
            }

        # Build context from retrieved chunks
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        context = "\n\n---\n\n".join(documents)

        # Generate response using RAG
        answer = await self._generate_rag_response(
            question=question, context=context, history=history, temperature=temperature
        )

        # Prepare sources
        sources = []
        for i, meta in enumerate(metadatas):
            sources.append(
                {
                    "file": meta.get("file_name", "Unknown"),
                    "chunk_index": meta.get("chunk_index", i),
                    "similarity": (
                        round(1 - distances[i], 3) if i < len(distances) else None
                    ),
                }
            )

        return {
            "success": True,
            "answer": answer,
            "data": None,
            "sources": sources,
            "error": None,
        }

    async def _generate_rag_response(
        self, question: str, context: str, history: List[Dict], temperature: float
    ) -> str:
        """Generate response for document RAG query"""

        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Only answer based on the provided context
2. If the context doesn't contain the answer, say so honestly
3. Be concise but thorough
4. Quote relevant parts when appropriate
5. If asked to summarize, provide key points"""

        # Build history context
        history_text = ""
        if history:
            history_parts = []
            for msg in history[-4:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content'][:200]}")
            history_text = (
                "Previous conversation:\n" + "\n".join(history_parts) + "\n\n"
            )

        user_prompt = f"""{history_text}Context from documents:
{context}

Question: {question}

Please answer based on the context provided."""

        try:
            response = client.chat.completions.create(
                model=getattr(settings, "llm_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1000,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"I found relevant documents but encountered an error generating the response: {str(e)}"


# Singleton instance
query_service = QueryService()
