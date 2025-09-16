import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    SPACY_MODEL_PATH = os.getenv("SPACY_MODEL_PATH", r"C:\Users\KNPRO\Desktop\projects\legal_analyzer\legal_NER\models\model-best")
    LEGAL_BERT_PATH = os.getenv("LEGAL_BERT_PATH", r"C:\Users\KNPRO\Desktop\projects\legal_analyzer\legalbert")
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # CORS settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]
    
    # File upload settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx'}
    
    # RAG settings
    RAG_INDEX_PATH = "./data/rag_index"
    RAG_CHUNK_SIZE = 400
    RAG_CHUNK_OVERLAP = 100

settings = Settings()
