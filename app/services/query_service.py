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
            top_k=min(top_k, 3),  # Usually 2-3 tables are enough
        )

        if not relevant_tables:
            return {
                "success": False,
                "answer": "No relevant Excel data found for your question.",
                "data": None,
                "sql": None,
                "error": "No matching tables",
            }

        # Step 2: Get detailed schema prompts from SQLite (more reliable than stored text)
        for t in relevant_tables:
            schema_prompt = sqlite_store.get_table_schema_for_prompt(t["table_name"])
            if schema_prompt:
                t["prompt_text"] = schema_prompt

        # Step 3: Get sample data for context
        sample_data = {}
        for table in relevant_tables[:2]:
            sample_df = sqlite_store.get_sample_data(table["table_name"], 3)
            if sample_df is not None:
                sample_data[table["table_name"]] = sample_df.to_string(index=False)

        # Step 3: Generate SQL
        sql, explanation = sql_generator.generate_sql(
            question=question,
            schemas=relevant_tables,
            sample_data=sample_data,
            conversation_history=history,
            temperature=temperature,
        )

        if not sql:
            return {
                "success": False,
                "answer": "I couldn't generate a query for your question. Could you try rephrasing it?",
                "data": None,
                "sql": None,
                "error": explanation,
            }

        # Step 4: Execute with retry
        result_df, final_sql, error = self._execute_with_retry(
            sql=sql, user_id=user_id, schemas=relevant_tables
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

        # Step 5: Generate natural language response
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
                "rows": result_df.head(100).to_dict("records"),  # Limit rows
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
        self, sql: str, user_id: str, schemas: List[Dict]
    ) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
        """Execute SQL with automatic retry on failure"""

        current_sql = sql
        schema_context = "\n\n".join([s.get("prompt_text", str(s)) for s in schemas])

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
