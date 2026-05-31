from fastapi import APIRouter, HTTPException

from rag import get_rag_response

router = APIRouter()


@router.get("/query/")
async def query_rag_system(q: str):
    try:
        response = get_rag_response(q)
        return {"query": q, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
