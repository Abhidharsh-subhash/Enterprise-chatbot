from typing import List
from pypdf import PdfReader
from docx import Document
import pandas as pd


def extract_text_from_file(file_path: str) -> str:
    # 4. EXCEL (Optimized for Vector DB / Embeddings)
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        try:
            # Read all sheets
            all_sheets = pd.read_excel(file_path, sheet_name=None)
            text_parts = []

            for sheet_name, df in all_sheets.items():
                if df.empty:
                    continue

                # Convert all data to string and handle NaN
                df = df.fillna("").astype(str)

                text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")

                # Method: Convert each row into a sentence/context string
                # This ensures that if the text is chunked, the context is preserved.
                for _, row in df.iterrows():
                    # Creates a string like: "Column1: Value1 | Column2: Value2 | ..."
                    row_text = " | ".join(
                        [f"{col}: {val}" for col, val in row.items() if val]
                    )
                    text_parts.append(row_text)

            return "\n".join(text_parts)

        except Exception as e:
            raise ValueError(f"Error processing Excel file: {e}")

    else:
        raise ValueError("Unsupported file format")


def split_text_into_chunks(
    text: str, chunk_size: int = 800, overlap: int = 100
) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
