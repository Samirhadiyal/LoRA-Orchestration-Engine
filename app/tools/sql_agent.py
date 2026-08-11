import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from app.database.engine import engine
from app.llm.client import DEFAULT_MODEL, llm_client

SQL_TRANSLATION_PROMPT = """
You are an expert PostgreSQL DBA for NeuroMesh.
Your job is to translate a natural language request into a valid, safe, read-only SQL query.

Target Database Schema:
1. `chat_sessions` table:
   - `id`: UUID (Primary Key)
   - `title`: VARCHAR / TEXT
   - `created_at`: TIMESTAMP / DATETIME
   - `updated_at`: TIMESTAMP / DATETIME

RULES:
- Respond ONLY with a valid JSON object containing the key 'sql_query'.
- Query MUST be read-only (SELECT queries ONLY). No INSERT, UPDATE, DELETE, DROP, or ALTER.
- Do NOT wrap SQL in markdown code blocks inside the JSON string.

Example Output:
{
  "sql_query": "SELECT COUNT(*) FROM chat_sessions;"
}
"""

class SQLAgent:
    """
    Translates natural language questions to PostgreSQL queries and executes them.
    """

    async def _translate_to_sql(self, query: str) -> str:
        """Uses LLM to turn natural language into read-only SQL."""
        response = await llm_client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SQL_TRANSLATION_PROMPT},
                {"role": "user", "content": f"User Request: {query}"}
            ],
            temperature=0.0
        )
        
        content = json.loads(response.choices[0].message.content)
        return content.get("sql_query", "")

    async def execute_query(self, natural_language_query: str) -> dict[str, Any]:
        """
        1. Translates text to SQL via LLM.
        2. Executes query via SQLModel engine against Postgres.
        3. Returns formatted result dictionary.
        """
        try:
            # 1. Translate
            sql_query = await self._translate_to_sql(natural_language_query)
            
            # Security guardrail: Ensure read-only query
            cleaned_sql = sql_query.strip().upper()
            if not cleaned_sql.startswith("SELECT") and not cleaned_sql.startswith("WITH"):
                return {
                    "success": False,
                    "error": "Security Error: Only read-only SELECT queries are allowed.",
                    "sql_generated": sql_query
                }

            # 2. Execute against Postgres
            with Session(engine) as session:
                result = session.exec(text(sql_query))
                
                # Fetch row records
                if result.returns_rows:
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                else:
                    rows = []

            # 3. Return response dict
            return {
                "success": True,
                "sql_generated": sql_query,
                "row_count": len(rows),
                "data": rows
            }

        except (json.JSONDecodeError, SQLAlchemyError, ValueError) as e:
            return {
                "success": False,
                "error": str(e),
                "sql_generated": sql_query if 'sql_query' in locals() else None
            }