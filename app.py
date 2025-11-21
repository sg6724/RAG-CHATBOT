"""
api.py - FastAPI Server
Handles: HTTP endpoints for document upload and querying
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil

# Import our modules
import embeddings
import rag

# Initialize FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents and query with RAG",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 FastAPI server initialized!")


@app.get("/")
def root():
    """Root endpoint - API info"""
    return {
        "message": "RAG Chatbot API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload_text": "/upload/text",
            "upload_file": "/upload/file",
            "upload_url": "/upload/url",
            "query": "/query",
            "clear": "/clear"
        }
    }


@app.get("/health")
def health_check():
    """Health check - returns database stats"""
    try:
        count = embeddings.get_document_count()
        return {
            "status": "healthy",
            "documents_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/text")
def upload_text(text: str = Form(...), source_name: str = Form(...)):
    """
    Upload plain text
    
    Body (form-data):
        - text: Text content
        - source_name: Name for this text
    """
    try:
        chunks_count = embeddings.add_text(text, source_name)
        return {
            "status": "success",
            "chunks_created": chunks_count,
            "source": source_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/file")
def upload_file(file: UploadFile = File(...)):
    """
    Upload document file (PDF, DOCX, TXT)
    
    Body (multipart/form-data):
        - file: Document file
    """
    try:
        # Validate file type
        allowed_extensions = ["pdf", "docx", "txt"]
        file_ext = file.filename.split(".")[-1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {allowed_extensions}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Process file
        chunks_count = embeddings.add_file(tmp_path, file.filename)
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "chunks_created": chunks_count,
            "filename": file.filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/url")
def upload_url(url: str = Form(...)):
    """
    Upload content from URL
    
    Body (form-data):
        - url: URL to scrape
    """
    try:
        chunks_count = embeddings.add_url(url)
        return {
            "status": "success",
            "chunks_created": chunks_count,
            "url": url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")


@app.post("/query")
def query(question: str = Form(...), source_filter: str = Form(None)):
    """
    Query the RAG system
    
    Body (form-data):
        - question: User's question
        - source_filter: Optional - filter by specific source
    """
    try:
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if source_filter:
            result = rag.query_with_filter(question, source_filter)
        else:
            result = rag.query_documents(question)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to get response from RAG system")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.delete("/clear")
def clear_database():
    """Clear all documents from database"""
    try:
        embeddings.clear_database()
        return {
            "status": "success",
            "message": "Database cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
