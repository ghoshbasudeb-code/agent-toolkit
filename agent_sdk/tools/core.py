import os
import json
import requests
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

@tool
def calculate_metrics(data_points: List[float]) -> dict:
    """Calculates mean and total for a list of numeric values."""
    if not data_points:
        return {"mean": 0.0, "total": 0.0}
    return {
        "mean": sum(data_points) / len(data_points),
        "total": sum(data_points),
    }


@tool
def format_json_response(data: dict) -> str:
    """Formats a dictionary into a clean JSON string representation."""
    return json.dumps(data, indent=2)


@tool
def search_web(query: str) -> str:
    """Searches the live web for recent information or news using Tavily."""
    search = TavilySearchResults(max_results=3)
    results = search.invoke({"query": query})
    return str(results)


@tool
def fetch_api_data(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Performs an HTTP GET request to fetch data from an external REST API endpoint."""
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"API request failed: {str(e)}"


@tool
def execute_sql_query(query: str, db_uri: str = "sqlite:///example.db") -> str:
    """Executes a read-only SQL query against a database and returns the results."""
    if not query.strip().lower().startswith("select"):
        return "Error: Only read-only SELECT queries are allowed."

    try:
        engine = create_engine(db_uri)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()
            keys = result.keys()
            output = [dict(zip(keys, row)) for row in rows]
            return str(output)
    except SQLAlchemyError as e:
        return f"Database query failed: {str(e)}"

@tool
def query_knowledge_base(query: str, db_path: str = "vectorstore") -> str:
    """Queries a local vector database knowledge base to retrieve relevant context from custom documents."""
    if not os.path.exists(db_path):
        return f"Error: Vector store directory '{db_path}' not found."

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(query, k=3)

        if not docs:
            return "No relevant information found in the knowledge base."

        # Format retrieved chunks
        context = "\n\n".join(
            [f"--- Chunk {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs)]
        )
        return context
    except Exception as e:
        return f"Failed to retrieve context: {str(e)}"
