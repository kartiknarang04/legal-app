from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import tempfile
import asyncio
from pathlib import Path
import logging

# Import our model classes
from models.ner_model import LegalNERProcessor
from models.summarizer_model import LegalDocumentSummarizer
from models.advanced_rag_model import SessionBasedAdvancedRAG
from config import settings

from utils.document_processor import document_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Document Analyzer API",
    description="API for legal document analysis using NER, summarization, and RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
ner_processor = None
summarizer = None
rag_system = None

# Pydantic models for API
class DocumentAnalysisRequest(BaseModel):
    text: str
    summary_length: Optional[int] = 5
    use_groq_refinement: Optional[bool] = True

class RAGQueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # Added session_id parameter
    top_k: Optional[int] = 5
    use_rerank: Optional[bool] = True

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None  # Added session_id to response
    analysis: Optional[Dict[str, Any]] = None
    document_processing: Optional[Dict[str, Any]] = None  # Added document_processing to response

class RAGQueryResponse(BaseModel):
    success: bool
    answer: str
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CreateSessionRequest(BaseModel):
    session_name: Optional[str] = None

class SessionResponse(BaseModel):
    success: bool
    session_id: str
    session_info: Dict[str, Any]

class AddDocumentToSessionRequest(BaseModel):
    text: str
    title: Optional[str] = ""
    session_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global ner_processor, summarizer, rag_system
    
    try:
        logger.info("Initializing models...")
        
        spacy_model_path = settings.SPACY_MODEL_PATH
        legal_bert_path = settings.LEGAL_BERT_PATH
        groq_api_key = settings.GROQ_API_KEY
        
        # Initialize NER processor
        logger.info("Loading NER model...")
        ner_processor = LegalNERProcessor(spacy_model_path)
        
        # Initialize summarizer
        logger.info("Loading summarization model...")
        summarizer = LegalDocumentSummarizer(legal_bert_path, groq_api_key)
        
        logger.info("Loading Session-based Advanced RAG system...")
        rag_system = SessionBasedAdvancedRAG(
            model_path=legal_bert_path,
            groq_api_key=groq_api_key,
            index_path=settings.RAG_INDEX_PATH,
            use_colbert=True
        )
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        # Continue startup even if models fail to load
        pass

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Legal Document Analyzer API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    status = {
        "api": "healthy",
        "models": {
            "ner": ner_processor is not None,
            "summarizer": summarizer is not None,
            "rag": rag_system is not None
        }
    }
    return status

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), session_id: Optional[str] = None):
    """
    Upload and analyze a legal document with OCR support
    Performs NER and summarization automatically
    Creates new session if none provided
    """
    supported_extensions = ('.txt', '.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    if not file.filename.lower().endswith(supported_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Please upload {', '.join(supported_extensions)} files."
        )
    
    # Check file size (limit to 50MB)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
    
    try:
        # Read file content
        content = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        processing_result = document_processor.process_document(file.filename, content)
        
        if not processing_result['success']:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process document: {processing_result.get('error', 'Unknown error')}"
            )
        
        text = processing_result['text']
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or contains no readable text")
        
        logger.info(f"Extracted text length: {len(text)} characters using {processing_result['extraction_method']}")
        
        if rag_system:
            actual_session_id = rag_system.add_documents_to_session(
                documents=[{
                    "text": text, 
                    "title": file.filename,
                    "metadata": {
                        "extraction_method": processing_result['extraction_method'],
                        "confidence": processing_result['confidence'],
                        "pages_processed": processing_result['pages_processed'],
                        "ocr_pages": processing_result['ocr_pages'],
                        "file_size": processing_result['file_size']
                    }
                }],
                session_id=session_id
            )
            session_id = actual_session_id
            logger.info(f"Added document to RAG session {session_id}")
        
        analysis_result = await analyze_document_async(text)
        
        analysis_result['document_processing'] = {
            'extraction_method': processing_result['extraction_method'],
            'confidence': processing_result['confidence'],
            'pages_processed': processing_result['pages_processed'],
            'ocr_pages': processing_result['ocr_pages'],
            'file_type': processing_result['file_type']
        }
        
        return DocumentUploadResponse(
            success=True,
            message=f"Document '{file.filename}' analyzed successfully using {processing_result['extraction_method']}",
            session_id=session_id,
            analysis=analysis_result,
            document_processing=processing_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/analyze")
async def analyze_text(request: DocumentAnalysisRequest):
    """
    Analyze text directly (without file upload)
    """
    try:
        analysis_result = await analyze_document(
            request.text, 
            request.summary_length, 
            request.use_groq_refinement
        )
        
        return {
            "success": True,
            "analysis": analysis_result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

async def analyze_document_async(text: str, summary_length: int = 5, use_groq: bool = True) -> Dict[str, Any]:
    """
    Perform complete document analysis (NER + Summarization) asynchronously
    """
    result = {
        "ner_results": None,
        "summary_results": None,
        "text_stats": {
            "character_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(text.split('.'))
        }
    }
    
    import asyncio
    
    async def run_ner():
        if ner_processor:
            try:
                logger.info("Starting NER analysis...")
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                ner_results = await loop.run_in_executor(None, ner_processor.process_text, text)
                logger.info("NER analysis completed")
                return ner_results
            except Exception as e:
                logger.error(f"NER analysis failed: {str(e)}")
                return {"error": str(e)}
        return None
    
    async def run_summarization():
        if summarizer:
            try:
                logger.info("Starting summarization...")
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                summary_results = await loop.run_in_executor(
                    None, 
                    summarizer.summarize, 
                    text, 
                    summary_length, 
                    use_groq
                )
                logger.info("Summarization completed")
                return summary_results
            except Exception as e:
                logger.error(f"Summarization failed: {str(e)}")
                return {"error": str(e)}
        return None
    
    # Run both tasks concurrently
    ner_task = run_ner()
    summary_task = run_summarization()
    
    ner_results, summary_results = await asyncio.gather(ner_task, summary_task)
    
    result["ner_results"] = ner_results
    result["summary_results"] = summary_results
    
    return result

async def analyze_document(text: str, summary_length: int = 5, use_groq: bool = True) -> Dict[str, Any]:
    """
    Perform complete document analysis (NER + Summarization)
    """
    return await analyze_document_async(text, summary_length, use_groq)

@app.post("/rag/add_document")
async def add_document_to_rag(request: DocumentAnalysisRequest):
    """
    Add a document to the RAG system for future querying
    Creates new session if none exists
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        session_id = rag_system.create_session("Document Analysis Session")
        
        # Add document to RAG index
        documents = [{
            "text": request.text,
            "title": f"Document_{len(request.text.split())} words"
        }]
        
        rag_system.add_documents_to_session(documents, session_id=session_id)
        
        return {
            "success": True,
            "message": "Document added to new RAG session",
            "session_id": session_id,
            "total_documents": len(rag_system.sessions[session_id]['documents'])
        }
        
    except Exception as e:
        logger.error(f"Error adding document to RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding document to RAG: {str(e)}")

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """
    Query the Advanced RAG system for answers
    Uses session-based search with multi-stage retrieval
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        logger.info(f"Processing RAG query: {request.question} (Session: {request.session_id})")
        
        if request.session_id:
            result = rag_system.query_session(
                request.question,
                session_id=request.session_id,
                top_k=request.top_k
            )
        else:
            # Fallback to general query
            result = rag_system.query(
                request.question,
                top_k=request.top_k
            )
        
        if 'error' in result:
            return RAGQueryResponse(
                success=False,
                answer=f"Error: {result['error']}",
                contexts=[],
                metadata={"error": result['error']}
            )
        
        contexts = []
        for source in result.get('sources', []):
            contexts.append({
                'text': source.get('excerpt', ''),
                'title': source.get('title', ''),
                'score': source.get('relevance_score', 0.0),  # Use 'score' for frontend compatibility
                'relevance_score': source.get('relevance_score', 0.0),  # Keep original field too
                'section': source.get('section', ''),
                'entities': source.get('entities', []),
                'chunk_index': source.get('chunk_index', 0),
                'legal_terms': source.get('legal_terms', []),
                'relevant_passages': source.get('relevant_passages', [])
            })
        
        return RAGQueryResponse(
            success=True,
            answer=result.get('answer', 'No answer generated'),
            contexts=contexts,
            metadata={
                'query_type': result.get('query_analysis', {}).get('type', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'entities_found': result.get('query_analysis', {}).get('entities', []),
                'key_concepts': result.get('query_analysis', {}).get('concepts', []),
                'expanded_queries': result.get('query_analysis', {}).get('expanded_queries', []),
                'session_id': result.get('session_id'),
                'session_info': result.get('session_info', {}),
                'num_contexts': len(contexts),
                'model_used': 'SessionBasedAdvancedRAG'
            }
        )
        
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying RAG: {str(e)}")

@app.get("/rag/status")
async def rag_status():
    """
    Get RAG system status
    """
    if not rag_system:
        return {"available": False, "sessions": 0, "total_documents": 0}
    
    sessions = rag_system.list_sessions()
    total_docs = sum(session.get('document_count', 0) for session in sessions)
    
    return {
        "available": True,
        "sessions": len(sessions),
        "total_documents": total_docs,
        "session_list": sessions
    }

@app.post("/sessions/create", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new RAG session"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        session_id = rag_system.create_session(request.session_name)
        session_info = rag_system.get_session_info(session_id)
        
        return SessionResponse(
            success=True,
            session_id=session_id,
            session_info=session_info
        )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all available sessions"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        sessions = rag_system.list_sessions()
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        session_info = rag_system.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session_info": session_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")

@app.post("/sessions/{session_id}/documents")
async def add_document_to_session(session_id: str, request: AddDocumentToSessionRequest):
    """Add a document to a specific session"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        documents = [{"text": request.text, "title": request.title}]
        actual_session_id = rag_system.add_documents_to_session(
            documents, 
            session_id=session_id
        )
        
        session_info = rag_system.get_session_info(actual_session_id)
        
        return {
            "success": True,
            "message": "Document added to session",
            "session_id": actual_session_id,
            "session_info": session_info
        }
    except Exception as e:
        logger.error(f"Error adding document to session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding document to session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
