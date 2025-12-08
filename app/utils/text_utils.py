from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document
import pandas as pd


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from files.
    NOTE: For Excel files, this is only used for non-SQLite approach.
    The SQLite approach uses extract_excel_schema() instead.
    """

    if file_path.endswith(".pdf"):
        return _extract_from_pdf(file_path)

    elif file_path.endswith(".docx"):
        return _extract_from_docx(file_path)

    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        # This is the OLD approach - keeping for backwards compatibility
        # New approach uses SQLite directly
        return _extract_from_excel_legacy(file_path)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def _extract_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _extract_from_docx(file_path: str) -> str:
    """Extract text from Word document"""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def _extract_from_excel_legacy(file_path: str) -> str:
    """
    Legacy Excel extraction (converts to text).
    NOT recommended for analytics - use SQLite approach instead.
    """
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        text_parts = []

        for sheet_name, df in all_sheets.items():
            if df.empty:
                continue

            df = df.fillna("").astype(str)
            text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")

            for _, row in df.iterrows():
                row_text = " | ".join(
                    [f"{col}: {val}" for col, val in row.items() if val]
                )
                text_parts.append(row_text)

        return "\n".join(text_parts)

    except Exception as e:
        raise ValueError(f"Error processing Excel file: {e}")


def split_text_into_chunks(
    text: str, chunk_size: int = 800, overlap: int = 100
) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ============================================
# NEW FUNCTIONS FOR EXCEL SCHEMA
# ============================================


def create_schema_embedding_text(schema_info: Dict[str, Any]) -> str:
    parts = [
        f"Excel spreadsheet from file {schema_info['original_filename']}",
        f"sheet named {schema_info['sheet_name']}",
        f"containing {schema_info['row_count']} rows of data.",
        f"The data is stored in SQLite table {schema_info['table_name']}.",
        "The table has the following columns:",
    ]

    for col in schema_info["columns"]:
        logical = col.get("logical_type", "").lower()
        col_type_desc = logical or col["type"].lower()

        col_text = (
            f"{col['name']} (original header '{col['original_name']}') "
            f"storing {col_type_desc} values"
        )

        if col.get("samples"):
            samples = [str(s) for s in col["samples"][:3]]
            col_text += f" such as {', '.join(samples)}"

        if col.get("stats") and col["stats"].get("min") is not None:
            col_text += f" ranging from {col['stats']['min']} to {col['stats']['max']}"

        if col.get("logical_type") == "date":
            col_text += " in YYYY-MM-DD format"

        parts.append(col_text)

    return ". ".join(parts)


def is_excel_file(file_path: str) -> bool:
    """Check if file is an Excel file"""
    return file_path.lower().endswith((".xlsx", ".xls"))
