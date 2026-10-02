import os
import torch
from typing import Optional
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Singleton model and Pinecone index instances for efficiency
_MODEL: Optional[SentenceTransformer] = None
_INDEX = None


def _get_model_and_index(index_name: str = "dl-ai"):
    """Lazy loader for SentenceTransformer and Pinecone Index."""
    global _MODEL, _INDEX

    if _MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    if _INDEX is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")

        pc = Pinecone(api_key=api_key)
        _INDEX = pc.Index(index_name)

    return _MODEL, _INDEX


@tool
def pinecone_semantic_search(query: str, top_k: int = 5) -> str:
    """Performs semantic vector search against a Pinecone index containing 
    Quora duplicate questions using SentenceTransformers (all-MiniLM-L6-v2)."""
    try:
        model, index = _get_model_and_index()

        # Generate embedding vector
        embedding = model.encode(query).tolist()

        # Execute vector query against Pinecone
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )

        matches = results.get("matches", [])
        if not matches:
            return "No matching records found in vector index."

        # Format retrieved matches for the agent's context window
        formatted_matches = []
        for i, match in enumerate(matches, 1):
            score = round(match["score"], 3)
            text = match.get("metadata", {}).get("text", "N/A")
            formatted_matches.append(f"{i}. [Score: {score}] {text}")

        return "\n".join(formatted_matches)

    except Exception as e:
        return f"Semantic search query failed: {str(e)}"
