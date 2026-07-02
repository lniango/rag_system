from fastapi import APIRouter, HTTPException
from app.rag import get_rag_response

# Create API router to manage API routes
router = APIRouter()

@router.get("/query/")
async def query_rag_system(query : str):
    
    
    try:
        # Pass the query string to the RAG system and return the response    
        response = await get_rag_response(query)
        
        return {
            "query": query, 
            "response": response['result'] if isinstance(response, dict) else str(response)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))