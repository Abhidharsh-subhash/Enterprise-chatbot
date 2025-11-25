import os
import pandas as pd
import traceback
from app.core.openai_client import client, CHAT_MODEL
from app.core.logger import logger

UPLOADS_DIR = os.path.abspath("./uploads")


def execute_pandas_retrieval(file_name: str, query: str) -> str:
    """
    1. Loads the full Excel file.
    2. Asks LLM to write Python code to answer the specific query.
    3. Executes code to get EITHER an analytic number OR specific row details.
    4. Returns the raw string result.
    """
    file_path = os.path.join(UPLOADS_DIR, file_name)

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError("The source Excel file could not be found.")

    try:
        # 1. Load Data
        df = pd.read_excel(file_path)

        # Clean column names (remove spaces/special chars for easier coding)
        original_columns = list(df.columns)
        df.columns = [str(col).strip() for col in df.columns]

        # 2. Generate Code Prompt
        # We give the LLM the schema and the query
        schema_info = df.dtypes.to_string()

        prompt = f"""
        You are a Python Data Analyst. 
        I have a DataFrame `df` with columns: {list(df.columns)}.
        Column Types:
        {schema_info}
        
        User Query: "{query}"
        
        Write Python code to extract the exact answer.
        - If the user asks for a count/sum/avg -> Calculate it.
        - If the user asks for details (e.g. 'Order status for ID 123') -> Filter the DataFrame and show the relevant row(s).
        
        RULES:
        1. Store the final output in a variable named `result`.
        2. If `result` is a DataFrame or Series, convert it to a string using .to_string() or .to_json().
        3. Do NOT generate charts or plots.
        4. Return ONLY valid Python code. No markdown markers.
        """

        # 3. Get Code from LLM
        response = client.chat.completions.create(
            model=CHAT_MODEL,  # GPT-4 recommended for accurate coding
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python coding assistant. Output only code.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        code = (
            response.choices[0]
            .message.content.replace("```python", "")
            .replace("```", "")
            .strip()
        )
        logger.info(f"Generated Pandas Code: {code}")

        # 4. Execute Code
        local_vars = {"df": df, "pd": pd}

        try:
            exec(code, {}, local_vars)
        except Exception as exec_err:
            # Retry logic could go here, but for now return error
            return f"Error executing data logic: {exec_err}"

        # 5. Retrieve Result
        raw_result = local_vars.get(
            "result", "No result variable was created by the code."
        )

        return str(raw_result)

    except Exception as e:
        logger.error(f"Pandas Wrapper Error: {traceback.format_exc()}")
        return f"Could not process Excel file: {str(e)}"
