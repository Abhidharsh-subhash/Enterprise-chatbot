import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re
from datetime import datetime, timezone

from app.core.config import settings


class SQLiteStore:
    """Manages SQLite database for Excel data storage"""

    def __init__(self):
        self.db_path = Path(settings.sqlite_path) / settings.sqlite_database
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_metadata_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_metadata_table(self):
        """Create metadata table to track all imported tables"""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _excel_tables_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT UNIQUE NOT NULL,
                    original_filename TEXT NOT NULL,
                    sheet_name TEXT NOT NULL,
                    columns_info TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def import_excel(
        self, file_path: str, user_id: str, doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Import Excel file into SQLite.
        Returns list of created tables with their schema info.
        """
        file_path = Path(file_path)
        excel_file = pd.ExcelFile(file_path)
        created_tables = []

        for sheet_name in excel_file.sheet_names:
            # Read sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            if df.empty:
                continue

            # Clean column names for SQL
            df.columns = self._clean_column_names(df.columns)

            # Generate unique table name (include user_id to avoid conflicts)
            table_name = self._generate_table_name(file_path.stem, sheet_name, user_id)

            # Extract column info BEFORE storing
            columns_info = self._extract_columns_info(df)

            # Store data in SQLite
            with self._get_connection() as conn:
                # Drop if exists (for re-uploads)
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

                # Create table and insert data
                df.to_sql(table_name, conn, index=False, if_exists="replace")

                # Store metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO _excel_tables_metadata 
                    (table_name, original_filename, sheet_name, columns_info, 
                     row_count, user_id, doc_id, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        table_name,
                        file_path.name,
                        sheet_name,
                        json.dumps(columns_info),
                        len(df),
                        str(user_id),
                        doc_id,
                        str(file_path),
                    ),
                )
                conn.commit()

            # Prepare schema info for embedding
            schema_info = {
                "table_name": table_name,
                "original_filename": file_path.name,
                "sheet_name": sheet_name,
                "columns": columns_info,
                "row_count": len(df),
                "user_id": str(user_id),
                "doc_id": doc_id,
            }

            created_tables.append(schema_info)

        return created_tables

    def _clean_column_names(self, columns) -> List[str]:
        """Make column names SQL-friendly"""
        clean_names = []
        seen = set()

        for col in columns:
            clean = str(col).strip().lower()
            clean = re.sub(r"[^a-z0-9_]", "_", clean)
            clean = re.sub(r"_+", "_", clean).strip("_")

            if not clean:
                clean = "column"
            if clean[0].isdigit():
                clean = "col_" + clean

            # Handle duplicates
            original = clean
            counter = 1
            while clean in seen:
                clean = f"{original}_{counter}"
                counter += 1

            seen.add(clean)
            clean_names.append(clean)

        return clean_names

    def _generate_table_name(self, filename: str, sheet_name: str, user_id: str) -> str:
        """Generate unique table name"""
        # Use last 8 chars of user_id for uniqueness
        user_suffix = str(user_id)[-8:].replace("-", "")

        clean_filename = re.sub(r"[^a-z0-9]", "_", filename.lower())
        clean_sheet = re.sub(r"[^a-z0-9]", "_", sheet_name.lower())

        table_name = f"t_{user_suffix}_{clean_filename}_{clean_sheet}"
        table_name = re.sub(r"_+", "_", table_name).strip("_")

        return table_name[:60]

    def _extract_columns_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract detailed column information for schema"""
        columns_info = []

        for col in df.columns:
            col_data = df[col]

            # Determine type
            if pd.api.types.is_integer_dtype(col_data):
                col_type = "INTEGER"
            elif pd.api.types.is_float_dtype(col_data):
                col_type = "DECIMAL"
            elif pd.api.types.is_bool_dtype(col_data):
                col_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_type = "DATETIME"
            else:
                col_type = "TEXT"

            # Get sample values (only 5, not all data!)
            samples = col_data.dropna().unique()[:5].tolist()
            samples = [str(s)[:50] for s in samples]  # Limit length too

            # Basic stats for numeric columns
            stats = {}
            if pd.api.types.is_numeric_dtype(col_data) and not col_data.isna().all():
                stats = {
                    "min": float(col_data.min()) if pd.notna(col_data.min()) else None,
                    "max": float(col_data.max()) if pd.notna(col_data.max()) else None,
                }

            columns_info.append(
                {
                    "name": col,
                    "type": col_type,
                    "samples": samples,
                    "null_count": int(col_data.isna().sum()),
                    "unique_count": int(col_data.nunique()),
                    "stats": stats,
                }
            )

        return columns_info

    def execute_query(
        self, sql: str, user_id: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query safely.
        Returns (result_df, error_message)
        """
        # Security: Basic SQL validation
        sql_upper = sql.upper().strip()
        dangerous = [
            "DROP",
            "DELETE",
            "TRUNCATE",
            "ALTER",
            "INSERT",
            "UPDATE",
            "CREATE",
        ]

        for keyword in dangerous:
            if keyword in sql_upper:
                if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
                    return None, f"Operation not allowed: {keyword}"

        # Security: Verify user can only access their tables
        # Extract table names from query and verify ownership
        user_tables = self.get_user_tables(user_id)
        user_table_names = [t["table_name"] for t in user_tables]

        # Simple check - in production, use proper SQL parsing
        for table in user_table_names:
            if table in sql:
                break
        else:
            # No user table found in query - might be trying to access others' data
            if user_table_names:  # Only check if user has tables
                return None, "Access denied: You can only query your own tables"

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(sql, conn)
                return df, None
        except Exception as e:
            return None, str(e)

    def get_user_tables(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all tables for a specific user"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT table_name, original_filename, sheet_name, 
                       columns_info, row_count, doc_id, created_at
                FROM _excel_tables_metadata
                WHERE user_id = ?
                ORDER BY created_at DESC
            """,
                (str(user_id),),
            )

            tables = []
            for row in cursor.fetchall():
                tables.append(
                    {
                        "table_name": row[0],
                        "original_filename": row[1],
                        "sheet_name": row[2],
                        "columns": json.loads(row[3]),
                        "row_count": row[4],
                        "doc_id": row[5],
                        "created_at": row[6],
                    }
                )

            return tables

    def get_table_schema_for_prompt(self, table_name: str) -> Optional[str]:
        """Get formatted schema for LLM prompt"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT table_name, original_filename, sheet_name, 
                       columns_info, row_count
                FROM _excel_tables_metadata
                WHERE table_name = ?
            """,
                (table_name,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            columns = json.loads(row[3])

            lines = [
                f"Table: {row[0]}",
                f"Source: {row[1]} (Sheet: {row[2]})",
                f"Total Rows: {row[4]}",
                "",
                "Columns:",
            ]

            for col in columns:
                line = f"  - {col['name']} ({col['type']})"
                if col["samples"]:
                    samples_str = ", ".join(col["samples"][:3])
                    line += f" | Examples: {samples_str}"
                if col["stats"]:
                    if col["stats"].get("min") is not None:
                        line += (
                            f" | Range: {col['stats']['min']} to {col['stats']['max']}"
                        )
                lines.append(line)

            return "\n".join(lines)

    def get_sample_data(
        self, table_name: str, limit: int = 5
    ) -> Optional[pd.DataFrame]:
        """Get sample rows from a table"""
        try:
            with self._get_connection() as conn:
                return pd.read_sql_query(
                    f"SELECT * FROM {table_name} LIMIT {limit}", conn
                )
        except:
            return None

    def delete_user_tables(self, user_id: str, doc_id: str = None):
        """Delete tables for a user (optionally for specific doc_id)"""
        with self._get_connection() as conn:
            if doc_id:
                # Delete specific document's tables
                cursor = conn.execute(
                    """
                    SELECT table_name FROM _excel_tables_metadata
                    WHERE user_id = ? AND doc_id = ?
                """,
                    (str(user_id), doc_id),
                )
            else:
                # Delete all user's tables
                cursor = conn.execute(
                    """
                    SELECT table_name FROM _excel_tables_metadata
                    WHERE user_id = ?
                """,
                    (str(user_id),),
                )

            for row in cursor.fetchall():
                conn.execute(f"DROP TABLE IF EXISTS {row[0]}")

            if doc_id:
                conn.execute(
                    """
                    DELETE FROM _excel_tables_metadata
                    WHERE user_id = ? AND doc_id = ?
                """,
                    (str(user_id), doc_id),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM _excel_tables_metadata
                    WHERE user_id = ?
                """,
                    (str(user_id),),
                )

            conn.commit()


# Singleton instance
sqlite_store = SQLiteStore()
