# Legal Document Analyzer

A comprehensive full-stack application for analyzing legal documents using advanced AI models including Named Entity Recognition (NER), Document Summarization, and Retrieval-Augmented Generation (RAG).

## Features

- **Named Entity Recognition**: Extract and categorize legal entities (persons, organizations, courts, statutes, etc.)
- **Document Summarization**: Generate extractive and AI-refined summaries with customizable length
- **RAG Question Answering**: Ask natural language questions about your documents
- **File Upload Support**: Process .txt, .pdf, and .docx files
- **Interactive UI**: Modern, responsive interface built with Next.js and Tailwind CSS

## Architecture

- **Frontend**: Next.js 15 with TypeScript, Tailwind CSS, and shadcn/ui components
- **Backend**: FastAPI with Python, integrating spaCy, LegalBERT, and Groq
- **Models**: 
  - spaCy NER model for entity extraction
  - LegalBERT for document summarization
  - Sentence Transformers + FAISS for RAG retrieval
  - Groq API for answer generation and summary refinement

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Your trained models:
  - spaCy NER model
  - LegalBERT model
- Groq API key (optional, for enhanced features)

### Installation

1. **Clone and install dependencies:**
   \`\`\`bash
   git clone <your-repo>
   cd legal-document-analyzer
   npm run setup
   \`\`\`

2. **Configure your models:**
   \`\`\`bash
   # Copy environment template
   cp backend/.env.example backend/.env
   
   # Edit backend/.env with your model paths and API keys
   SPACY_MODEL_PATH=./models/spacy_model
   LEGAL_BERT_PATH=./models/legalbert
   GROQ_API_KEY=your_groq_api_key_here
   \`\`\`

3. **Place your models:**
   \`\`\`
   backend/
   ├── models/
   │   ├── spacy_model/          # Your trained spaCy NER model
   │   └── legalbert/            # Your LegalBERT model
   \`\`\`

### Development

**Start both frontend and backend with one command:**
\`\`\`bash
npm run dev
\`\`\`

This will start:
- Frontend at http://localhost:3000
- Backend API at http://localhost:8000

**Individual commands:**
\`\`\`bash
# Frontend only
npm run dev:frontend

# Backend only
npm run dev:backend

# Install backend dependencies
npm run backend:install
\`\`\`

### Docker Deployment

**Build and run with Docker Compose:**
\`\`\`bash
docker-compose up --build
\`\`\`

**Or build individual containers:**
\`\`\`bash
# Build frontend
docker build -f Dockerfile.frontend -t legal-analyzer-frontend .

# Build backend
docker build -f Dockerfile.backend -t legal-analyzer-backend .
\`\`\`

## API Endpoints

### Document Analysis
- `POST /upload` - Upload and analyze a document file
- `POST /analyze` - Analyze text directly
- `GET /health` - Health check with model status

### RAG System
- `POST /rag/add_document` - Add document to RAG knowledge base
- `POST /rag/query` - Query the RAG system
- `GET /rag/status` - Get RAG system status

## Usage Guide

### 1. Upload & Analyze Documents
- Upload legal documents (.txt, .pdf, .docx)
- Automatic NER and summarization processing
- View extracted entities grouped by type
- Customize summary length and AI refinement

### 2. Named Entity Recognition
- View all extracted entities with categories
- See entity statistics and repetition rates
- Browse entities by type (Person, Organization, Court, etc.)

### 3. Document Summarization
- Generate extractive summaries
- Use AI refinement for improved readability
- Adjust summary length (3-20 sentences)
- Compare extractive vs. refined summaries

### 4. RAG Question Answering
- Add documents to knowledge base
- Ask natural language questions
- View source documents and relevance scores
- See legal terms and relevant passages

## Model Integration

### spaCy NER Model
\`\`\`python
# Your NER model should be loadable with:
nlp = spacy.load("path/to/your/model")

# Expected entity types:
# PERSON, ORG, COURT, JUDGE, LAWYER, STATUTE, PRECEDENT, etc.
\`\`\`

### LegalBERT Model
\`\`\`python
# Your LegalBERT model should be compatible with:
tokenizer = AutoTokenizer.from_pretrained("path/to/legalbert")
model = AutoModel.from_pretrained("path/to/legalbert")
\`\`\`

### RAG System
- Uses Sentence Transformers for embeddings
- FAISS for vector similarity search
- Optional LegalBERT reranking
- Groq API for answer generation

## Configuration

### Environment Variables
\`\`\`bash
# Model paths
SPACY_MODEL_PATH=./models/spacy_model
LEGAL_BERT_PATH=./models/legalbert

# API keys
GROQ_API_KEY=your_groq_api_key

# Server settings
HOST=0.0.0.0
PORT=8000
\`\`\`

### Customization
- Modify entity colors in `components/ner-results.tsx`
- Adjust summarization weights in `backend/models/summarizer_model.py`
- Configure RAG parameters in `backend/models/rag_model.py`

## Troubleshooting

### Common Issues

1. **Models not loading:**
   - Check model paths in `.env`
   - Ensure models are compatible with the expected format
   - Check Python dependencies

2. **Backend connection errors:**
   - Verify backend is running on port 8000
   - Check CORS settings in `backend/main.py`
   - Ensure all Python dependencies are installed

3. **File upload issues:**
   - Check file size limits (10MB default)
   - Verify supported file types (.txt, .pdf, .docx)
   - For PDF/DOCX, implement proper parsing in `backend/utils/file_processor.py`

### Development Tips

- Use `npm run dev` for hot reloading on both frontend and backend
- Check browser console and backend logs for errors
- Test API endpoints directly at http://localhost:8000/docs
- Use the health endpoint to verify model loading

## Production Deployment

### Docker Production
\`\`\`bash
# Build production images
docker-compose -f docker-compose.prod.yml up --build

# Or deploy to cloud platforms
# Configure environment variables for production
\`\`\`

### Manual Deployment
1. Build frontend: `npm run build`
2. Set up Python environment with requirements
3. Configure reverse proxy (nginx)
4. Set production environment variables
5. Use process manager (PM2, systemd)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Open an issue on GitHub
