from fastapi import APIRouter, Query
from typing import Dict, Any
from app.retrieval.search import hybrid_search

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., description="The search prompt or question"),
    limit: int = Query(default=5, ge=1, le=20, description="Max number of chunks to return")
) -> Dict[str, Any]:
    """
    Performs Hybrid Search (Dense Vectors + BM25 + Cross-Encoder Reranking).
    """
    results = hybrid_search(query=query, top_k=limit)
    return {
        "query": query,
        "count": len(results),
        "results": results
    }