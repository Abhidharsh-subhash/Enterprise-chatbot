# app/utils/column_analyzer.py
from typing import Dict, List, Optional, Any
import pandas as pd
from app.vector_store.sqlite_store import sqlite_store


class ColumnAnalyzer:
    """Analyze columns to provide context for SQL generation"""

    # Columns that typically have categorical/enum values
    CATEGORICAL_INDICATORS = [
        "service",
        "type",
        "status",
        "category",
        "method",
        "mode",
        "level",
        "tier",
        "plan",
        "grade",
        "rating",
        "priority",
        "department",
        "region",
        "country",
        "state",
        "city",
        "gender",
        "role",
        "position",
        "channel",
        "source",
    ]

    # Date column semantic mappings
    DATE_COLUMN_SEMANTICS = {
        "created": "when the record/order was created or initiated",
        "placed": "when the order was placed by customer",
        "ordered": "when the order was placed by customer",
        "closed": "when the order was completed/closed",
        "completed": "when the order was completed",
        "delivered": "when the order was delivered",
        "shipped": "when the order was shipped",
        "paid": "when the payment was made",
        "payment": "when the payment was made",
        "updated": "when the record was last updated",
        "modified": "when the record was last modified",
        "cancelled": "when the order was cancelled",
        "refunded": "when the refund was processed",
        "scheduled": "when the service is scheduled",
        "due": "when the payment/delivery is due",
        "start": "when something started",
        "end": "when something ended",
        "expiry": "when something expires",
    }

    def __init__(self, max_distinct_values: int = 20):
        self.max_distinct_values = max_distinct_values

    def get_column_context(
        self, table_name: str, columns: List[Dict], user_id: str
    ) -> Dict[str, Any]:
        """
        Get rich context about columns including distinct values and semantics.
        """
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

            # Identify and describe date columns
            if any(date_kw in col_lower for date_kw in ["date", "time", "_at", "_on"]):
                semantic = self._get_date_semantic(col_name)
                context["date_columns"][col_name] = {
                    "type": col_type,
                    "semantic": semantic,
                }

            # Identify categorical columns and get distinct values
            elif self._is_likely_categorical(col_name, col_type):
                distinct_values = self._get_distinct_values(
                    table_name, col_name, user_id
                )
                if distinct_values:
                    context["categorical_columns"][col_name] = {
                        "values": distinct_values,
                        "count": len(distinct_values),
                    }

            # Classify other columns
            elif col_type in ["INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"]:
                context["numeric_columns"].append(col_name)
            else:
                context["text_columns"].append(col_name)

        return context

    def _is_likely_categorical(self, col_name: str, col_type: str) -> bool:
        """Check if column is likely categorical based on name"""
        col_lower = col_name.lower().replace("_", " ")

        # Check against known categorical indicators
        for indicator in self.CATEGORICAL_INDICATORS:
            if indicator in col_lower:
                return True

        # Also check if it ends with common categorical suffixes
        categorical_suffixes = [
            "_type",
            "_status",
            "_category",
            "_service",
            "_method",
            "_mode",
        ]
        for suffix in categorical_suffixes:
            if col_name.lower().endswith(suffix):
                return True

        return False

    def _get_date_semantic(self, col_name: str) -> str:
        """Get semantic meaning of a date column"""
        col_lower = col_name.lower()

        for keyword, semantic in self.DATE_COLUMN_SEMANTICS.items():
            if keyword in col_lower:
                return semantic

        return "date/time field"

    def _get_distinct_values(
        self, table_name: str, column_name: str, user_id: str
    ) -> Optional[List[str]]:
        """Get distinct values for a column"""
        try:
            query = f"""
                SELECT DISTINCT "{column_name}" 
                FROM "{table_name}" 
                WHERE "{column_name}" IS NOT NULL 
                LIMIT {self.max_distinct_values + 1}
            """

            result_df, error = sqlite_store.execute_query(query, user_id)

            if error or result_df is None or len(result_df) == 0:
                return None

            values = result_df[column_name].tolist()

            # If too many values, it's probably not categorical
            if len(values) > self.max_distinct_values:
                return None

            return [str(v) for v in values if v is not None]

        except Exception as e:
            return None

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """Format column context for LLM prompt"""
        parts = []

        # Date columns with semantics
        if context["date_columns"]:
            parts.append("\n**DATE COLUMNS (with semantic meaning):**")
            for col_name, info in context["date_columns"].items():
                parts.append(f"  - {col_name}: {info['semantic']}")

        # Categorical columns with values
        if context["categorical_columns"]:
            parts.append("\n**CATEGORICAL COLUMNS (with possible values):**")
            for col_name, info in context["categorical_columns"].items():
                values_str = ", ".join([f"'{v}'" for v in info["values"][:10]])
                if info["count"] > 10:
                    values_str += f" ... and {info['count'] - 10} more"
                parts.append(f"  - {col_name}: [{values_str}]")

        return "\n".join(parts)


# Singleton instance
column_analyzer = ColumnAnalyzer()
