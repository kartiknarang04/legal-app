import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple, Optional
import faiss
import pickle
import os
from dataclasses import dataclass, field
import json
from pathlib import Path
import hashlib
from tqdm import tqdm
from groq import Groq
import re
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import uuid
from datetime import datetime

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@dataclass
class Document:
    """Represents a legal document or chunk"""
    id: str
    text: str
    title: str = ""
    session_id: str = ""  # Added session_id to track document sessions
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
@dataclass
class RetrievedContext:
    """Represents retrieved context with scores"""
    document: Document
    score: float
    relevant_passages: List[str] = field(default_factory=list)

class SessionRAGSystem:
    """Manages session-based RAG operations"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 legal_bert_path: str = None,
                 groq_api_key: str = None,
                 chunk_size: int = 400,
                 chunk_overlap: int = 100,
                 index_path: str = None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load sentence transformer for embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.sentence_model = SentenceTransformer(embedding_model)
        self.sentence_model.to(self.device)
        
        self.legal_bert = None
        self.legal_tokenizer = None
        if legal_bert_path and os.path.exists(legal_bert_path):
            print("Loading Legal-BERT for reranking...")
            self.legal_tokenizer = AutoTokenizer.from_pretrained(legal_bert_path, local_files_only=True)
            self.legal_bert = AutoModel.from_pretrained(legal_bert_path, local_files_only=True)
            self.legal_bert.to(self.device)
            self.legal_bert.eval()
        
        # Initialize Groq
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        self.index_path = index_path
        
        # Session-based storage
        self.sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session data
        self.current_session_id: Optional[str] = None
    
    def create_session(self, session_name: str = None) -> str:
        """Create a new session for document indexing"""
        session_id = str(uuid.uuid4())
        session_name = session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.sessions[session_id] = {
            'name': session_name,
            'documents': [],
            'index': None,
            'created_at': datetime.now().isoformat(),
            'doc_word_counts': [],
            'vocabulary': set(),
            'idf_scores': {}
        }
        
        self.current_session_id = session_id
        print(f"Created new session: {session_name} (ID: {session_id})")
        return session_id
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False
    
    def get_session_info(self, session_id: str = None) -> Dict[str, Any]:
        """Get information about a session"""
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'name': session['name'],
            'document_count': len(session['documents']),
            'created_at': session['created_at'],
            'has_index': session['index'] is not None
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions"""
        return [self.get_session_info(sid) for sid in self.sessions.keys()]

    def smart_chunk_text(self, text: str, title: str = "", session_id: str = "") -> List[Document]:
        """
        Intelligently chunk text preserving legal document structure
        """
        # Clean text while preserving structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r' +', ' ', text)  # Remove extra spaces
        
        chunks = []
        
        # Try to detect legal document structure
        section_pattern = r'(\d+\.[\s\S]*?)(?=\d+\.|$)'
        sections = re.findall(section_pattern, text)
        
        if not sections:
            # Fall back to paragraph-based chunking
            paragraphs = text.split('\n\n')
            current_chunk = []
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = len(self.sentence_model.tokenizer.tokenize(para))
                
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, title, len(chunks), session_id))
                    
                    # Handle overlap
                    if self.chunk_overlap > 0 and len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]  # Keep last paragraph
                        current_tokens = len(self.sentence_model.tokenizer.tokenize(current_chunk[0]))
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(para)
                current_tokens += para_tokens
            
            # Add remaining
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, title, len(chunks), session_id))
        else:
            # Chunk by sections
            for section in sections:
                section_tokens = len(self.sentence_model.tokenizer.tokenize(section))
                
                if section_tokens <= self.chunk_size:
                    chunks.append(self._create_chunk(section, title, len(chunks), session_id))
                else:
                    # Split large sections into sentences
                    sentences = sent_tokenize(section)
                    current_chunk = []
                    current_tokens = 0
                    
                    for sent in sentences:
                        sent_tokens = len(self.sentence_model.tokenizer.tokenize(sent))
                        
                        if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append(self._create_chunk(chunk_text, title, len(chunks), session_id))
                            
                            # Overlap
                            overlap_sents = max(1, len(current_chunk) // 4)
                            current_chunk = current_chunk[-overlap_sents:]
                            current_tokens = sum(len(self.sentence_model.tokenizer.tokenize(s)) 
                                               for s in current_chunk)
                        
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
                    
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk(chunk_text, title, len(chunks), session_id))
        
        # Update total chunks metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _create_chunk(self, text: str, title: str, index: int, session_id: str) -> Document:
        """Create a document chunk with metadata and session ID"""
        chunk_id = hashlib.md5(f"{session_id}_{title}_{index}_{text[:50]}".encode()).hexdigest()[:16]
        
        # Extract key legal terms for better retrieval
        legal_terms = self._extract_legal_terms(text)
        
        return Document(
            id=chunk_id,
            text=text,
            title=title,
            session_id=session_id,  # Added session_id to document
            metadata={
                'chunk_index': index,
                'total_chunks': -1,
                'source_title': title,
                'legal_terms': legal_terms,
                'text_length': len(text)
            }
        )
    
    def add_documents_to_session(self, documents: List[Dict[str, str]], 
                                session_id: str = None, 
                                chunk_documents: bool = True) -> str:
        """
        Add documents to a specific session (creates new session if none provided)
        
        Args:
            documents: List of dicts with 'text' and optional 'title' keys
            session_id: Target session ID (creates new if None)
            chunk_documents: Whether to chunk documents
            
        Returns:
            Session ID where documents were added
        """
        # Create new session if none provided or doesn't exist
        if not session_id or session_id not in self.sessions:
            session_id = self.create_session()
        else:
            self.current_session_id = session_id
        
        session = self.sessions[session_id]
        new_docs = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            text = doc.get('text', '')
            title = doc.get('title', '')
            
            if chunk_documents:
                chunks = self.smart_chunk_text(text, title, session_id)
                new_docs.extend(chunks)
            else:
                new_docs.append(self._create_chunk(text, title, 0, session_id))
        
        if not new_docs:
            return session_id
        
        print(f"Adding {len(new_docs)} chunks to session {session_id}...")
        
        # Generate embeddings
        texts = [doc.text for doc in new_docs]
        embeddings = self.get_embeddings(texts)
        
        # Assign embeddings
        for doc, emb in zip(new_docs, embeddings):
            doc.embedding = emb
        
        # Add to session document store
        session['documents'].extend(new_docs)
        
        # Build session-specific indices
        self._build_session_faiss_index(session_id)
        self._build_session_bm25_index(session_id)
        
        print(f"Session {session_id} now contains {len(session['documents'])} chunks")
        return session_id
    
    def _build_session_faiss_index(self, session_id: str) -> None:
        """Build FAISS index for a specific session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        documents = session['documents']
        
        if not documents:
            return
        
        embeddings = np.vstack([doc.embedding for doc in documents])
        
        # Use L2 normalization + Inner Product = Cosine Similarity
        if len(documents) < 10000:
            # For small datasets, use flat index
            index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # For larger datasets, use IVF index for faster search
            nlist = min(100, len(documents) // 100)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.train(embeddings.astype('float32'))
        
        index.add(embeddings.astype('float32'))
        session['index'] = index
    
    def _build_session_bm25_index(self, session_id: str) -> None:
        """Build BM25 components for a specific session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        documents = session['documents']
        
        if not documents:
            return
        
        print(f"Building BM25 index for session {session_id}...")
        
        # Tokenize all documents in this session
        doc_word_counts = []
        vocabulary = set()
        
        for doc in documents:
            words = word_tokenize(doc.text.lower())
            word_count = {}
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
                vocabulary.add(word)
            doc_word_counts.append(word_count)
        
        # Calculate IDF scores for this session
        num_docs = len(documents)
        idf_scores = {}
        
        for word in vocabulary:
            doc_freq = sum(1 for wc in doc_word_counts if word in wc)
            idf_scores[word] = np.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        
        # Store in session
        session['doc_word_counts'] = doc_word_counts
        session['vocabulary'] = vocabulary
        session['idf_scores'] = idf_scores

    def search_session(self, query: str, session_id: str = None, top_k: int = 5, 
                      rerank: bool = True, use_hybrid: bool = True) -> List[RetrievedContext]:
        """
        Search within a specific session only
        
        Args:
            query: Search query
            session_id: Session to search in (uses current if None)
            top_k: Number of documents to retrieve
            rerank: Whether to rerank results
            use_hybrid: Use hybrid search
            
        Returns:
            List of retrieved contexts from the session
        """
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.sessions:
            print("No valid session for search!")
            return []
        
        session = self.sessions[session_id]
        documents = session['documents']
        index = session['index']
        
        if not index or not documents:
            print(f"No documents indexed in session {session_id}!")
            return []
        
        # Semantic search with FAISS
        query_embedding = self.get_embeddings([query], show_progress=False)
        query_embedding = query_embedding.astype('float32')
        
        # Retrieve more candidates for reranking
        search_k = min(top_k * 3, len(documents))
        scores, indices = index.search(query_embedding, search_k)
        
        # Create initial contexts
        contexts = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(documents):
                contexts.append(RetrievedContext(
                    document=documents[idx],
                    score=float(score)
                ))
        
        # Hybrid search: combine with BM25 for this session
        if use_hybrid and session['doc_word_counts']:
            bm25_scores = [self._session_bm25_score(query, i, session_id) for i in range(len(documents))]
            bm25_top_indices = np.argsort(bm25_scores)[-search_k:][::-1]
            
            # Merge scores
            seen_ids = {ctx.document.id for ctx in contexts}
            for idx in bm25_top_indices:
                if documents[idx].id not in seen_ids:
                    contexts.append(RetrievedContext(
                        document=documents[idx],
                        score=bm25_scores[idx] * 0.3  # Weight BM25 lower
                    ))
        
        # Rerank if requested
        if rerank and self.legal_bert:
            contexts = self._legal_bert_rerank(query, contexts, top_k * 2)
        
        # Final filtering and passage extraction
        contexts = contexts[:top_k]
        for context in contexts:
            context.relevant_passages = self._extract_relevant_passages(
                query, context.document.text
            )
        
        return contexts
    
    def _session_bm25_score(self, query: str, doc_idx: int, session_id: str, 
                           k1: float = 1.2, b: float = 0.75) -> float:
        """Calculate BM25 score for a document within a session"""
        session = self.sessions[session_id]
        query_words = word_tokenize(query.lower())
        doc_word_count = session['doc_word_counts'][doc_idx]
        
        # Calculate average document length for this session
        avg_doc_len = np.mean([sum(wc.values()) for wc in session['doc_word_counts']])
        doc_len = sum(doc_word_count.values())
        
        score = 0.0
        for word in query_words:
            if word not in session['vocabulary']:
                continue
            
            tf = doc_word_count.get(word, 0)
            idf = session['idf_scores'].get(word, 0)
            
            numerator = idf * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            
            score += numerator / denominator
        
        return score

    def generate_answer(self, query: str, contexts: List[RetrievedContext],
                       system_prompt: str = None, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Generate answer with improved prompt engineering
        """
        if not self.groq_client:
            return {
                'answer': 'Groq client not initialized. Please provide API key.',
                'contexts_used': [],
                'success': False
            }
        
        if not contexts:
            return {
                'answer': 'No relevant documents found for your query.',
                'contexts_used': [],
                'success': False
            }
        
        # Prepare context with better formatting
        context_texts = []
        total_tokens = 0
        max_tokens_per_context = 2000
        
        for i, ctx in enumerate(contexts, 1):
            # Use relevant passages if available, otherwise full text
            if ctx.relevant_passages:
                text = ' '.join(ctx.relevant_passages)
            else:
                text = ctx.document.text
            
            # Truncate if necessary but preserve complete sentences
            if len(text) > max_tokens_per_context:
                sentences = sent_tokenize(text)
                truncated = []
                current_len = 0
                for sent in sentences:
                    if current_len + len(sent) > max_tokens_per_context:
                        break
                    truncated.append(sent)
                    current_len += len(sent)
                text = ' '.join(truncated)
            
            context_entry = (
                f"[Document {i}]\n"
                f"Title: {ctx.document.title or 'Legal Document'}\n"
                f"Relevance Score: {ctx.score:.2f}\n"
                f"Content: {text}\n"
            )
            
            context_texts.append(context_entry)
            total_tokens += len(text.split())
            
            # Limit total context to avoid token limits
            if total_tokens > 3000:
                break
        
        combined_context = "\n---\n".join(context_texts)
        
        # Enhanced system prompt for legal domain
        if not system_prompt:
            system_prompt = """You are an expert legal assistant with deep knowledge of law and legal procedures. 
Your role is to provide accurate, well-reasoned legal analysis based on the provided documents.

Guidelines:
1. Base your answer strictly on the provided documents
2. Cite specific document numbers when making claims
3. Use precise legal terminology
4. If the documents don't contain enough information, clearly state this
5. Highlight any relevant legal principles, sections, or precedents mentioned
6. Structure your response clearly with proper legal reasoning"""
        
        # Improved user prompt
        user_prompt = f"""Question: {query}

Available Legal Documents:
{combined_context}

Instructions:
1. Analyze the provided documents carefully
2. Answer the question based ONLY on the information in these documents
3. Cite document numbers [Document X] for each point you make
4. If the answer is not in the documents, state this clearly
5. Be concise but thorough in your legal analysis

Your Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",  # Use larger model for better quality
                temperature=temperature,
                max_tokens=2000,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'contexts_used': [
                    {
                        'title': ctx.document.title,
                        'score': ctx.score,
                        'chunk_index': ctx.document.metadata.get('chunk_index', 0),
                        'legal_terms': ctx.document.metadata.get('legal_terms', []),
                        'relevant_passages': ctx.relevant_passages[:2] if ctx.relevant_passages else []
                    }
                    for ctx in contexts
                ],
                'success': True,
                'model_used': "llama-3.1-8b-instant",
                'num_contexts': len(context_texts)
            }
            
        except Exception as e:
            return {
                'answer': f'Error generating answer: {str(e)}',
                'contexts_used': [],
                'success': False,
                'error': str(e)
            }
    
    def query_session(self, question: str, session_id: str = None, top_k: int = 5, 
                     rerank: bool = True, verbose: bool = True,
                     temperature: float = 0.3) -> Dict[str, Any]:
        """
        Complete RAG pipeline for a specific session
        """
        session_id = session_id or self.current_session_id
        
        if verbose:
            print(f"\n🔍 Processing query in session {session_id}: {question}")
            print("-" * 50)
        
        # Retrieve relevant documents from session
        contexts = self.search_session(question, session_id, top_k=top_k, rerank=rerank)
        
        if verbose:
            print(f"\n📚 Retrieved {len(contexts)} relevant documents from session:")
            for i, ctx in enumerate(contexts, 1):
                print(f"  {i}. {ctx.document.title or 'Document'} "
                      f"(Chunk {ctx.document.metadata.get('chunk_index', 0)}, "
                      f"Score: {ctx.score:.3f})")
                if ctx.document.metadata.get('legal_terms'):
                    print(f"     Legal terms: {', '.join(ctx.document.metadata['legal_terms'][:5])}")
        
        # Generate answer
        result = self.generate_answer(question, contexts, temperature=temperature)
        
        # Add session info to result
        result['session_id'] = session_id
        result['session_info'] = self.get_session_info(session_id)
        
        if verbose and result['success']:
            print(f"\n✅ Answer generated using {result.get('model_used')}")
            print(f"   Used {result.get('num_contexts', 0)} context chunks from session")
        
        return result

    def get_embeddings(self, texts: List[str], batch_size: int = 32,
                      show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings using sentence transformer
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        # Use sentence transformer for better semantic search
        embeddings = self.sentence_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        
        return embeddings
    
    def _extract_legal_terms(self, text: str) -> List[str]:
        """Extract legal terms from text for enhanced retrieval"""
        legal_keywords = [
            'plaintiff', 'defendant', 'appellant', 'respondent', 'court', 
            'judgment', 'order', 'section', 'article', 'clause', 'provision',
            'held', 'ruled', 'decided', 'appeal', 'petition', 'writ'
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in legal_keywords if term in text_lower]
        
        # Also extract section references (e.g., "Section 420", "Article 21")
        section_refs = re.findall(r'(?:section|article|clause)\s+\d+[a-z]?', text_lower)
        found_terms.extend(section_refs)
        
        return list(set(found_terms))
    
    def _legal_bert_rerank(self, query: str, contexts: List[RetrievedContext], 
                           top_k: int) -> List[RetrievedContext]:
        """
        Rerank using Legal-BERT for domain-specific understanding
        """
        if not self.legal_bert:
            return contexts[:top_k]
        
        print("Reranking with Legal-BERT...")
        
        with torch.no_grad():
            # Get query embedding from Legal-BERT
            query_inputs = self.legal_tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            query_outputs = self.legal_bert(**query_inputs)
            query_emb = query_outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
            # Score each context
            for ctx in contexts:
                doc_inputs = self.legal_tokenizer(
                    ctx.document.text[:512],  # Limit length
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                doc_outputs = self.legal_bert(**doc_inputs)
                doc_emb = doc_outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                
                # Combine original score with Legal-BERT similarity
                legal_sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                ctx.score = 0.6 * ctx.score + 0.4 * legal_sim
        
        contexts.sort(key=lambda x: x.score, reverse=True)
        return contexts[:top_k]
    
    def _extract_relevant_passages(self, query: str, document: str, 
                                  max_passages: int = 3) -> List[str]:
        """
        Extract most relevant passages with improved scoring
        """
        sentences = sent_tokenize(document)
        if len(sentences) <= max_passages:
            return sentences
        
        # Score sentences
        query_emb = self.get_embeddings([query], show_progress=False)[0]
        sent_embs = self.get_embeddings(sentences, show_progress=False)
        
        similarities = [np.dot(query_emb, sent_emb) for sent_emb in sent_embs]
        
        # Get top passages with context
        top_indices = np.argsort(similarities)[-max_passages:][::-1]
        top_indices = sorted(top_indices)
        
        # Include surrounding context if beneficial
        expanded_indices = set(top_indices)
        for idx in top_indices:
            if idx > 0 and similarities[idx - 1] > 0.5:  # Include previous sentence if relevant
                expanded_indices.add(idx - 1)
            if idx < len(sentences) - 1 and similarities[idx + 1] > 0.5:  # Include next sentence
                expanded_indices.add(idx + 1)
        
        final_indices = sorted(list(expanded_indices))[:max_passages + 1]
        return [sentences[i] for i in final_indices]

class ImprovedLegalRAGSystem(SessionRAGSystem):
    """
    Backward compatible RAG system that now uses session-based indexing
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a default session for backward compatibility
        self.default_session_id = self.create_session("Default Session")
    
    def add_documents(self, documents: List[Dict[str, str]], 
                      chunk_documents: bool = True) -> None:
        """Add documents to default session (backward compatibility)"""
        self.add_documents_to_session(
            documents, 
            session_id=self.default_session_id, 
            chunk_documents=chunk_documents
        )
    
    def search(self, query: str, top_k: int = 5, 
              rerank: bool = True, use_hybrid: bool = None) -> List[RetrievedContext]:
        """Search in default session (backward compatibility)"""
        return self.search_session(
            query, 
            session_id=self.default_session_id, 
            top_k=top_k, 
            rerank=rerank, 
            use_hybrid=use_hybrid if use_hybrid is not None else True
        )
    
    def query(self, question: str, top_k: int = 5, 
             rerank: bool = True, verbose: bool = True,
             temperature: float = 0.3) -> Dict[str, Any]:
        """Query default session (backward compatibility)"""
        return self.query_session(
            question, 
            session_id=self.default_session_id, 
            top_k=top_k, 
            rerank=rerank, 
            verbose=verbose, 
            temperature=temperature
        )
    
    @property
    def documents(self):
        """Get documents from default session (backward compatibility)"""
        if self.default_session_id in self.sessions:
            return self.sessions[self.default_session_id]['documents']
        return []
    
    @property
    def index(self):
        """Get index from default session (backward compatibility)"""
        if self.default_session_id in self.sessions:
            return self.sessions[self.default_session_id]['index']
        return None
