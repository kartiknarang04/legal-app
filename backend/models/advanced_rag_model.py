import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple, Optional, Set
import faiss
from dataclasses import dataclass, field
import json
from pathlib import Path
import hashlib
from tqdm import tqdm
from groq import Groq
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import spacy
import pickle
from rank_bm25 import BM25Okapi

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@dataclass
class LegalEntity:
    """Represents legal entities extracted from text"""
    text: str
    entity_type: str  # case_citation, statute, party, judge, concept
    context: str
    importance: float = 1.0

@dataclass
class DocumentChunk:
    """Enhanced document chunk with multiple representations"""
    id: str
    text: str
    title: str
    metadata: Dict[str, Any]
    
    # Multiple embeddings
    dense_embedding: Optional[np.ndarray] = None
    sparse_embedding: Optional[Dict] = None
    
    # Legal-specific features
    legal_entities: List[LegalEntity] = field(default_factory=list)
    legal_concepts: List[str] = field(default_factory=list)
    citation_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Contextual information
    section_type: str = ""  # facts, holding, reasoning, dissent
    importance_score: float = 1.0
    
    # For hierarchical retrieval
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)

@dataclass 
class QueryAnalysis:
    """Analyzed query with intent and entities"""
    original_query: str
    query_type: str = ""  # factual, precedent, statute_interpretation, procedure
    legal_entities: List[LegalEntity] = field(default_factory=list)
    temporal_context: Optional[str] = None
    jurisdiction: Optional[str] = None
    expanded_queries: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)

class AdvancedLegalRAG:
    def __init__(self, 
                 model_path: str,
                 groq_api_key: str = None,
                 index_path: str = "./advanced_legal_index",
                 use_colbert: bool = True):
        """
        Initialize Advanced Legal RAG System with SOTA techniques
        
        Args:
            model_path: Path to Legal-BERT model
            groq_api_key: Groq API key
            index_path: Path for index storage
            use_colbert: Whether to use ColBERT-style late interaction
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load Legal-BERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Groq
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        
        # Initialize SpaCy for NER (you might need: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Storage
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        self.chunks: List[DocumentChunk] = []
        
        # Multiple index types
        self.dense_index = None
        self.bm25_index = None
        self.concept_graph = nx.Graph()
        
        # ColBERT-style token indices
        self.use_colbert = use_colbert
        self.token_index = None
        self.token_to_chunks = defaultdict(set)
        
        # Legal knowledge base
        self._initialize_legal_knowledge()
        
    def _initialize_legal_knowledge(self):
        """Initialize legal concepts and patterns"""
        
        # Legal concept hierarchies
        self.legal_concepts = {
            'liability': ['negligence', 'strict liability', 'vicarious liability', 'product liability'],
            'contract': ['breach', 'consideration', 'offer', 'acceptance', 'damages', 'specific performance'],
            'criminal': ['mens rea', 'actus reus', 'intent', 'malice', 'premeditation'],
            'procedure': ['jurisdiction', 'standing', 'statute of limitations', 'res judicata'],
            'evidence': ['hearsay', 'relevance', 'privilege', 'burden of proof', 'admissibility'],
            'constitutional': ['due process', 'equal protection', 'free speech', 'search and seizure']
        }
        
        # Query patterns for intent classification
        self.query_patterns = {
            'precedent': ['case', 'precedent', 'ruling', 'held', 'decision'],
            'statute_interpretation': ['statute', 'section', 'interpretation', 'meaning', 'definition'],
            'factual': ['what happened', 'facts', 'circumstances', 'events'],
            'procedure': ['how to', 'procedure', 'process', 'filing', 'requirements'],
            'comparison': ['difference', 'compare', 'versus', 'distinguish']
        }
        
        # Legal relationship patterns
        self.relationship_patterns = {
            'overrules': ['overrules', 'overruled', 'overruling'],
            'distinguishes': ['distinguishes', 'distinguished', 'distinguishable'],
            'follows': ['follows', 'following', 'pursuant to'],
            'cites': ['cites', 'citing', 'cited'],
            'modifies': ['modifies', 'modified', 'amends']
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Deep analysis of legal query to understand intent
        
        Args:
            query: User query
            
        Returns:
            Analyzed query with metadata
        """
        analysis = QueryAnalysis(original_query=query)
        
        # Classify query type
        query_lower = query.lower()
        for qtype, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis.query_type = qtype
                break
        else:
            analysis.query_type = 'general'
        
        # Extract legal entities using SpaCy + rules
        doc = self.nlp(query)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'LAW']:
                analysis.legal_entities.append(
                    LegalEntity(ent.text, ent.label_, query)
                )
        
        # Extract case citations (pattern matching)
        case_pattern = r'\b\d+\s+[A-Z][a-z]+\.?\s+\d+\b|\bv\.\s+[A-Z][a-z]+'
        citations = re.findall(case_pattern, query)
        for citation in citations:
            analysis.legal_entities.append(
                LegalEntity(citation, 'case_citation', query)
            )
        
        # Extract key legal concepts
        for concept_category, concepts in self.legal_concepts.items():
            for concept in concepts:
                if concept in query_lower:
                    analysis.key_concepts.append(concept)
        
        # Generate expanded queries using different strategies
        analysis.expanded_queries = self._generate_expanded_queries(analysis)
        
        return analysis
    
    def _generate_expanded_queries(self, analysis: QueryAnalysis) -> List[str]:
        """
        Generate multiple query variations for better retrieval
        
        Args:
            analysis: Query analysis
            
        Returns:
            List of expanded queries
        """
        expanded = [analysis.original_query]
        
        # 1. Concept expansion
        if analysis.key_concepts:
            # Find related concepts
            related_concepts = []
            for concept in analysis.key_concepts:
                for category, concepts in self.legal_concepts.items():
                    if concept in concepts:
                        related_concepts.extend([c for c in concepts if c != concept])
            
            if related_concepts:
                expanded.append(f"{analysis.original_query} {' '.join(related_concepts[:3])}")
        
        # 2. Query reformulation based on type
        if analysis.query_type == 'precedent':
            expanded.append(f"legal precedent case law {analysis.original_query}")
            expanded.append(f"court held ruling {analysis.original_query}")
        elif analysis.query_type == 'statute_interpretation':
            expanded.append(f"statutory interpretation meaning {analysis.original_query}")
            expanded.append(f"legislative intent purpose {analysis.original_query}")
        
        # 3. Hypothetical document generation (HyDE)
        hyde_doc = self._generate_hypothetical_document(analysis.original_query)
        if hyde_doc:
            expanded.append(hyde_doc)
        
        return expanded
    
    def _generate_hypothetical_document(self, query: str) -> Optional[str]:
        """
        Generate hypothetical answer document (HyDE technique)
        
        Args:
            query: Original query
            
        Returns:
            Hypothetical document text
        """
        if not self.groq_client:
            return None
        
        try:
            prompt = f"""Generate a brief hypothetical legal document excerpt that would answer this question: {query}
            
            Write it as if it's from an actual legal case or statute. Be specific and use legal language.
            Keep it under 100 words."""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a legal expert generating hypothetical legal text."}, {"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=150
            )
            
            return response.choices[0].message.content
        except:
            return None
    
    def chunk_document_hierarchical(self, text: str, title: str = "") -> List[DocumentChunk]:
        """
        Create hierarchical chunks with parent-child relationships
        
        Args:
            text: Document text
            title: Document title
            
        Returns:
            List of hierarchical chunks
        """
        chunks = []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Identify sections (common legal document structure)
        section_patterns = [
            (r'(?i)\bFACTS?\b[:\s]', 'facts'),
            (r'(?i)\bHOLDING\b[:\s]', 'holding'),
            (r'(?i)\bREASONING\b[:\s]', 'reasoning'),
            (r'(?i)\bDISSENT\b[:\s]', 'dissent'),
            (r'(?i)\bCONCLUSION\b[:\s]', 'conclusion'),
            (r'(?i)\bBACKGROUND\b[:\s]', 'background')
        ]
        
        sections = []
        for pattern, section_type in section_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                sections.append((match.start(), section_type))
        
        sections.sort(key=lambda x: x[0])
        
        # Create parent chunk (full document summary)
        parent_id = hashlib.md5(f"{title}_parent".encode()).hexdigest()[:16]
        parent_chunk = DocumentChunk(
            id=parent_id,
            text=text[:1000],  # First 1000 chars as summary
            title=title,
            metadata={'level': 'parent', 'type': 'full_document'},
            section_type='full'
        )
        chunks.append(parent_chunk)
        
        # Create section-based chunks
        sentences = sent_tokenize(text)
        current_section = 'introduction'
        section_sentences = []
        
        for sent in sentences:
            # Check if we've entered a new section
            sent_pos = text.find(sent)
            for pos, stype in sections:
                if sent_pos >= pos:
                    if section_sentences and current_section != stype:
                        # Create chunk for previous section
                        chunk_id = hashlib.md5(f"{title}_{current_section}_{len(chunks)}".encode()).hexdigest()[:16]
                        chunk = DocumentChunk(
                            id=chunk_id,
                            text=' '.join(section_sentences),
                            title=title,
                            metadata={'level': 'section', 'section': current_section},
                            section_type=current_section,
                            parent_chunk_id=parent_id
                        )
                        chunks.append(chunk)
                        parent_chunk.child_chunk_ids.append(chunk_id)
                        section_sentences = []
                    current_section = stype
            
            section_sentences.append(sent)
            
            # Create chunk if we have enough sentences
            if len(section_sentences) >= 5:
                chunk_id = hashlib.md5(f"{title}_{current_section}_{len(chunks)}".encode()).hexdigest()[:16]
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=' '.join(section_sentences),
                    title=title,
                    metadata={'level': 'section', 'section': current_section},
                    section_type=current_section,
                    parent_chunk_id=parent_id
                )
                chunks.append(chunk)
                parent_chunk.child_chunk_ids.append(chunk_id)
                section_sentences = []
        
        # Add remaining sentences
        if section_sentences:
            chunk_id = hashlib.md5(f"{title}_{current_section}_{len(chunks)}".encode()).hexdigest()[:16]
            chunk = DocumentChunk(
                id=chunk_id,
                text=' '.join(section_sentences),
                title=title,
                metadata={'level': 'section', 'section': current_section},
                section_type=current_section,
                parent_chunk_id=parent_id
            )
            chunks.append(chunk)
            parent_chunk.child_chunk_ids.append(chunk_id)
        
        # Extract legal entities for each chunk
        for chunk in chunks:
            chunk.legal_entities = self._extract_legal_entities(chunk.text)
            chunk.importance_score = self._calculate_chunk_importance(chunk)
        
        return chunks
    
    def _extract_legal_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities from text"""
        entities = []
        doc = self.nlp(text[:5000])  # Limit for performance
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'LAW', 'GPE']:
                entities.append(
                    LegalEntity(ent.text, ent.label_, text[:100], importance=1.0)
                )
        
        # Legal citations
        citation_pattern = r'\b\d+\s+[A-Z][a-z]+\.?\s+\d+\b'
        for match in re.finditer(citation_pattern, text):
            entities.append(
                LegalEntity(match.group(), 'case_citation', text[:100], importance=2.0)
            )
        
        # Statute references
        statute_pattern = r'§\s*\d+[\.\d]*|\bSection\s+\d+'
        for match in re.finditer(statute_pattern, text):
            entities.append(
                LegalEntity(match.group(), 'statute', text[:100], importance=1.5)
            )
        
        return entities
    
    def _calculate_chunk_importance(self, chunk: DocumentChunk) -> float:
        """Calculate importance score for chunk"""
        score = 1.0
        
        # Section type weights
        section_weights = {
            'holding': 2.0,
            'conclusion': 1.8,
            'reasoning': 1.5,
            'facts': 1.2,
            'dissent': 0.8
        }
        score *= section_weights.get(chunk.section_type, 1.0)
        
        # Entity importance
        if chunk.legal_entities:
            entity_score = sum(e.importance for e in chunk.legal_entities) / len(chunk.legal_entities)
            score *= (1 + entity_score * 0.5)
        
        # Legal concept density
        text_lower = chunk.text.lower()
        concept_count = sum(1 for concepts in self.legal_concepts.values() 
                          for concept in concepts if concept in text_lower)
        score *= (1 + concept_count * 0.1)
        
        return score
    
    def get_colbert_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Generate ColBERT-style token embeddings
        
        Args:
            text: Input text
            
        Returns:
            Token embeddings and token strings
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            
            # Normalize token embeddings
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
        
        # Get actual tokens (not subwords)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
        
        # Filter special tokens
        mask = inputs['attention_mask'][0].cpu().numpy()
        valid_tokens = []
        valid_embeddings = []
        
        for i, (token, m) in enumerate(zip(tokens, mask)):
            if m == 1 and token not in ['[CLS]', '[SEP]', '[PAD]']:
                valid_tokens.append(token)
                valid_embeddings.append(token_embeddings[0, i].cpu().numpy())
        
        if valid_embeddings:
            return np.vstack(valid_embeddings), valid_tokens
        return np.array([]), []
    
    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index documents with multiple retrieval methods
        
        Args:
            documents: List of documents to index
        """
        all_chunks = []
        
        print("Creating hierarchical chunks...")
        for doc in tqdm(documents):
            chunks = self.chunk_document_hierarchical(
                doc['text'], 
                doc.get('title', 'Untitled')
            )
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        print(f"Created {len(self.chunks)} chunks")
        
        # 1. Dense embeddings (BERT)
        print("Generating dense embeddings...")
        texts = [chunk.text for chunk in self.chunks]
        dense_embeddings = self._get_batch_embeddings(texts)
        
        for chunk, emb in zip(self.chunks, dense_embeddings):
            chunk.dense_embedding = emb
        
        # 2. Build FAISS index
        print("Building FAISS index...")
        embeddings_matrix = np.vstack([c.dense_embedding for c in self.chunks])
        self.dense_index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        self.dense_index.add(embeddings_matrix.astype('float32'))
        
        # 3. BM25 index for sparse retrieval
        print("Building BM25 index...")
        tokenized_corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 4. ColBERT-style token index
        if self.use_colbert:
            print("Building ColBERT token index...")
            self._build_colbert_index()
        
        # 5. Build concept graph
        print("Building legal concept graph...")
        self._build_concept_graph()
        
        print("Indexing complete!")
    
    def _build_colbert_index(self):
        """Build ColBERT-style inverted index for token-level matching"""
        self.token_to_chunks = defaultdict(set)
        
        for i, chunk in enumerate(self.chunks):
            token_embs, tokens = self.get_colbert_embeddings(chunk.text[:512])
            
            for token in tokens:
                # Store token to chunk mapping
                self.token_to_chunks[token.lower()].add(i)
    
    def _build_concept_graph(self):
        """Build graph of legal concept relationships"""
        self.concept_graph = nx.Graph()
        
        for chunk in self.chunks:
            # Add nodes for chunks
            self.concept_graph.add_node(
                chunk.id,
                type='chunk',
                text=chunk.text[:200],
                importance=chunk.importance_score
            )
            
            # Add edges between chunks that share entities
            for other_chunk in self.chunks:
                if chunk.id != other_chunk.id:
                    shared_entities = set(e.text for e in chunk.legal_entities) & \
                                    set(e.text for e in other_chunk.legal_entities)
                    if shared_entities:
                        self.concept_graph.add_edge(
                            chunk.id, 
                            other_chunk.id,
                            weight=len(shared_entities)
                        )
    
    def _get_batch_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Generate embeddings in batches"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                                     torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def multi_stage_retrieval(self, query_analysis: QueryAnalysis, 
                            top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Multi-stage retrieval combining multiple strategies
        
        Args:
            query_analysis: Analyzed query
            top_k: Number of documents to retrieve
            
        Returns:
            Ranked list of chunks with scores
        """
        candidates = {}
        
        # Stage 1: Dense retrieval with expanded queries
        print("Stage 1: Dense retrieval...")
        for query in query_analysis.expanded_queries[:3]:
            query_emb = self._get_batch_embeddings([query])[0]
            scores, indices = self.dense_index.search(
                query_emb.reshape(1, -1).astype('float32'), 
                top_k * 2
            )
            
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.chunks):
                    chunk_id = self.chunks[idx].id
                    if chunk_id not in candidates:
                        candidates[chunk_id] = {'chunk': self.chunks[idx], 'scores': {}}
                    candidates[chunk_id]['scores']['dense'] = float(score)
        
        # Stage 2: BM25 sparse retrieval
        print("Stage 2: Sparse retrieval...")
        query_tokens = query_analysis.original_query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
        
        for idx in top_bm25_indices:
            if idx < len(self.chunks):
                chunk_id = self.chunks[idx].id
                if chunk_id not in candidates:
                    candidates[chunk_id] = {'chunk': self.chunks[idx], 'scores': {}}
                candidates[chunk_id]['scores']['bm25'] = float(bm25_scores[idx])
        
        # Stage 3: Entity-based retrieval
        print("Stage 3: Entity-based retrieval...")
        if query_analysis.legal_entities:
            for entity in query_analysis.legal_entities:
                for chunk in self.chunks:
                    chunk_entities = [e.text.lower() for e in chunk.legal_entities]
                    if entity.text.lower() in chunk_entities:
                        if chunk.id not in candidates:
                            candidates[chunk.id] = {'chunk': chunk, 'scores': {}}
                        candidates[chunk.id]['scores']['entity'] = \
                            candidates[chunk.id]['scores'].get('entity', 0) + 1.0
        
        # Stage 4: Graph-based expansion
        print("Stage 4: Graph-based retrieval...")
        if candidates:
            # Find related chunks through concept graph
            seed_chunks = list(candidates.keys())[:5]
            for chunk_id in seed_chunks:
                if chunk_id in self.concept_graph:
                    neighbors = self.concept_graph.neighbors(chunk_id)
                    for neighbor in list(neighbors)[:3]:
                        for chunk in self.chunks:
                            if chunk.id == neighbor and neighbor not in candidates:
                                candidates[neighbor] = {'chunk': chunk, 'scores': {}}
                                candidates[neighbor]['scores']['graph'] = 0.5
        
        # Combine scores with learned weights
        print("Combining scores...")
        final_scores = []
        
        # Weights for different retrieval methods (can be learned)
        weights = {
            'dense': 0.35,
            'bm25': 0.25,
            'entity': 0.25,
            'graph': 0.15
        }
        
        for chunk_id, data in candidates.items():
            chunk = data['chunk']
            scores = data['scores']
            
            # Normalize and combine scores
            final_score = 0
            for method, weight in weights.items():
                if method in scores:
                    # Normalize score to [0, 1]
                    if method == 'dense':
                        normalized = (scores[method] + 1) / 2  # Cosine similarity [-1, 1] to [0, 1]
                    elif method == 'bm25':
                        normalized = min(scores[method] / 10, 1)  # Cap at 10
                    elif method == 'entity':
                        normalized = min(scores[method] / 3, 1)  # Cap at 3 entities
                    else:
                        normalized = scores[method]
                    
                    final_score += weight * normalized
            
            # Boost by chunk importance
            final_score *= chunk.importance_score
            
            # Boost by section relevance to query type
            if query_analysis.query_type == 'precedent' and chunk.section_type == 'holding':
                final_score *= 1.5
            elif query_analysis.query_type == 'factual' and chunk.section_type == 'facts':
                final_score *= 1.5
            
            final_scores.append((chunk, final_score))
        
        # Sort and return top-k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores[:top_k]
    
    def generate_answer_with_reasoning(self, 
                                      query: str,
                                      retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> Dict[str, Any]:
        """
        Generate answer with legal reasoning using Chain-of-Thought
        
        Args:
            query: Original query
            retrieved_chunks: Retrieved chunks with scores
            
        Returns:
            Answer with reasoning and citations
        """
        if not self.groq_client:
            return {'error': 'Groq client not initialized'}
        
        # Prepare context with structured information
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            entities = ', '.join([e.text for e in chunk.legal_entities[:3]])
            context_parts.append(f"""
Document {i} [{chunk.title}] - Relevance: {score:.2f}
Section Type: {chunk.section_type}
Key Entities: {entities}
Content: {chunk.text[:800]}
""")
        
        context = "\n---\n".join(context_parts)
        
        # Create structured prompt for legal reasoning
        system_prompt = """You are an expert legal analyst. Provide thorough legal analysis using the IRAC method:
1. ISSUE: Identify the legal issue(s)
2. RULE: State the applicable legal rules/precedents
3. APPLICATION: Apply the rules to the facts
4. CONCLUSION: Provide a clear conclusion

Always cite specific documents when making claims. Be precise and use legal terminology appropriately."""
        
        user_prompt = f"""Query: {query}

Retrieved Legal Documents:
{context}

Please provide a comprehensive legal analysis addressing the query. Use the IRAC method and cite the documents."""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # Extract citations and structure the response
            result = {
                'answer': answer,
                'query_analysis': {
                    'type': query_analysis.query_type if 'query_analysis' in locals() else 'general',
                    'entities': [e.text for e in query_analysis.legal_entities] if 'query_analysis' in locals() else [],
                    'concepts': query_analysis.key_concepts if 'query_analysis' in locals() else []
                },
                'sources': [],
                'confidence': 0.0
            }
            
            # Add source information
            for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                source = {
                    'document_id': chunk.id,
                    'title': chunk.title,
                    'section': chunk.section_type,
                    'relevance_score': float(score),
                    'excerpt': chunk.text[:200] + '...',
                    'entities': [e.text for e in chunk.legal_entities[:5]]
                }
                result['sources'].append(source)
            
            # Calculate confidence based on retrieval scores
            if retrieved_chunks:
                avg_score = sum(score for _, score in retrieved_chunks[:3]) / min(3, len(retrieved_chunks))
                result['confidence'] = min(avg_score * 100, 100)
            
            return result
            
        except Exception as e:
            return {
                'error': f'Error generating answer: {str(e)}',
                'sources': [{'chunk': c.text[:200], 'score': s} for c, s in retrieved_chunks[:3]]
            }
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main query interface for the RAG system
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Complete response with answer and metadata
        """
        print(f"\nProcessing query: {query}")
        
        # Analyze query
        print("Analyzing query...")
        query_analysis = self.analyze_query(query)
        print(f"Query type: {query_analysis.query_type}")
        print(f"Entities found: {[e.text for e in query_analysis.legal_entities]}")
        print(f"Key concepts: {query_analysis.key_concepts}")
        
        # Multi-stage retrieval
        print("\nPerforming multi-stage retrieval...")
        retrieved_chunks = self.multi_stage_retrieval(query_analysis, top_k)
        
        if not retrieved_chunks:
            return {
                'error': 'No relevant documents found',
                'query_analysis': {
                    'type': query_analysis.query_type,
                    'entities': [e.text for e in query_analysis.legal_entities],
                    'concepts': query_analysis.key_concepts
                }
            }
        
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Generate answer
        print("\nGenerating answer with legal reasoning...")
        result = self.generate_answer_with_reasoning(query, retrieved_chunks)
        
        # Add query analysis to result
        result['query_analysis'] = {
            'type': query_analysis.query_type,
            'entities': [e.text for e in query_analysis.legal_entities],
            'concepts': query_analysis.key_concepts,
            'expanded_queries': query_analysis.expanded_queries[:3]
        }
        
        return result
    
    def rerank_with_cross_encoder(self, 
                                 query: str, 
                                 chunks: List[DocumentChunk],
                                 top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Rerank chunks using cross-encoder approach
        
        Args:
            query: Query text
            chunks: Chunks to rerank
            top_k: Number of top chunks to return
            
        Returns:
            Reranked chunks with scores
        """
        if not chunks:
            return []
        
        reranked = []
        
        for chunk in chunks:
            # Create query-document pair
            pair_text = f"[CLS] {query} [SEP] {chunk.text[:400]} [SEP]"
            
            inputs = self.tokenizer(
                pair_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding as relevance score
                cls_embedding = outputs.last_hidden_state[0, 0]
                score = torch.sigmoid(cls_embedding.mean()).item()
            
            reranked.append((chunk, score))
        
        # Sort by score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def save_index(self):
        """Save all indices and data to disk"""
        save_path = self.index_path
        
        # Save chunks
        chunks_data = []
        for chunk in self.chunks:
            chunk_dict = {
                'id': chunk.id,
                'text': chunk.text,
                'title': chunk.title,
                'metadata': chunk.metadata,
                'section_type': chunk.section_type,
                'importance_score': chunk.importance_score,
                'parent_chunk_id': chunk.parent_chunk_id,
                'child_chunk_ids': chunk.child_chunk_ids,
                'legal_entities': [
                    {'text': e.text, 'type': e.entity_type, 'importance': e.importance}
                    for e in chunk.legal_entities
                ],
                'legal_concepts': chunk.legal_concepts
            }
            chunks_data.append(chunk_dict)
        
        with open(save_path / 'chunks.json', 'w') as f:
            json.dump(chunks_data, f)
        
        # Save FAISS index
        if self.dense_index:
            faiss.write_index(self.dense_index, str(save_path / 'dense.index'))
        
        # Save embeddings
        if self.chunks and self.chunks[0].dense_embedding is not None:
            embeddings = np.vstack([c.dense_embedding for c in self.chunks])
            np.save(save_path / 'embeddings.npy', embeddings)
        
        # Save concept graph
        if self.concept_graph:
            with open(save_path / 'concept_graph.pkl', 'wb') as f:
                pickle.dump(self.concept_graph, f)
        
        print(f"Index saved to {save_path}")
    
    def load_index(self):
        """Load indices and data from disk"""
        load_path = self.index_path
        
        # Load chunks
        with open(load_path / 'chunks.json', 'r') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for chunk_dict in chunks_data:
            chunk = DocumentChunk(
                id=chunk_dict['id'],
                text=chunk_dict['text'],
                title=chunk_dict['title'],
                metadata=chunk_dict['metadata'],
                section_type=chunk_dict.get('section_type', ''),
                importance_score=chunk_dict.get('importance_score', 1.0),
                parent_chunk_id=chunk_dict.get('parent_chunk_id'),
                child_chunk_ids=chunk_dict.get('child_chunk_ids', [])
            )
            
            # Reconstruct legal entities
            chunk.legal_entities = [
                LegalEntity(
                    text=e['text'],
                    entity_type=e['type'],
                    context='',
                    importance=e.get('importance', 1.0)
                )
                for e in chunk_dict.get('legal_entities', [])
            ]
            chunk.legal_concepts = chunk_dict.get('legal_concepts', [])
            
            self.chunks.append(chunk)
        
        # Load FAISS index
        if (load_path / 'dense.index').exists():
            self.dense_index = faiss.read_index(str(load_path / 'dense.index'))
        
        # Load embeddings
        if (load_path / 'embeddings.npy').exists():
            embeddings = np.load(load_path / 'embeddings.npy')
            for chunk, emb in zip(self.chunks, embeddings):
                chunk.dense_embedding = emb
        
        # Load concept graph
        if (load_path / 'concept_graph.pkl').exists():
            with open(load_path / 'concept_graph.pkl', 'rb') as f:
                self.concept_graph = pickle.load(f)
        
        # Rebuild BM25 index
        tokenized_corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        print(f"Index loaded from {load_path}")
        print(f"Loaded {len(self.chunks)} chunks")

class SessionBasedAdvancedRAG(AdvancedLegalRAG):
    """Session-based wrapper for Advanced Legal RAG"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sessions = {}
        self.current_session_id = None
    
    def create_session(self, session_name: str = None) -> str:
        """Create a new session for document isolation"""
        import uuid
        import pandas as pd
        session_id = str(uuid.uuid4())[:16]
        
        self.sessions[session_id] = {
            'name': session_name or f"Session_{len(self.sessions) + 1}",
            'created_at': str(pd.Timestamp.now()),
            'documents': [],
            'chunks': [],
            'dense_index': None,
            'bm25_index': None,
            'concept_graph': nx.Graph(),
            'document_count': 0
        }
        
        return session_id
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            # Load session data
            session_data = self.sessions[session_id]
            self.chunks = session_data['chunks']
            self.dense_index = session_data['dense_index']
            self.bm25_index = session_data['bm25_index']
            self.concept_graph = session_data['concept_graph']
            return True
        return False
    
    def add_documents_to_session(self, documents: List[Dict[str, str]], session_id: str = None) -> str:
        """Add documents to a specific session"""
        if session_id is None:
            session_id = self.create_session()
        elif session_id not in self.sessions:
            session_id = self.create_session()
        
        # Set current session
        self.set_current_session(session_id)
        
        # Store documents in session
        self.sessions[session_id]['documents'].extend(documents)
        self.sessions[session_id]['document_count'] = len(self.sessions[session_id]['documents'])
        
        # Index documents for this session
        self.index_documents(documents)
        
        # Save session data
        self.sessions[session_id]['chunks'] = self.chunks
        self.sessions[session_id]['dense_index'] = self.dense_index
        self.sessions[session_id]['bm25_index'] = self.bm25_index
        self.sessions[session_id]['concept_graph'] = self.concept_graph
        
        return session_id
    
    def query_session(self, query: str, session_id: str, top_k: int = 5) -> Dict[str, Any]:
        """Query a specific session"""
        if session_id not in self.sessions:
            return {'error': f'Session {session_id} not found'}
        
        # Set current session
        if not self.set_current_session(session_id):
            return {'error': f'Failed to load session {session_id}'}
        
        # Perform query on session data
        result = self.query(query, top_k)
        result['session_id'] = session_id
        result['session_info'] = {
            'name': self.sessions[session_id]['name'],
            'document_count': self.sessions[session_id]['document_count']
        }
        
        return result
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        sessions = []
        for session_id, session_data in self.sessions.items():
            sessions.append({
                'session_id': session_id,
                'name': session_data['name'],
                'created_at': session_data['created_at'],
                'document_count': session_data['document_count']
            })
        return sessions
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        if session_id not in self.sessions:
            return None
        
        session_data = self.sessions[session_id]
        return {
            'session_id': session_id,
            'name': session_data['name'],
            'created_at': session_data['created_at'],
            'document_count': session_data['document_count'],
            'documents': [doc.get('title', 'Untitled') for doc in session_data['documents']]
        }

# Example usage
def main():
    """Example usage of the Advanced Legal RAG system"""
    
    # Initialize the system
    from backend.config import settings
    rag = SessionBasedAdvancedRAG(
        model_path=settings.LEGAL_BERT_PATH,  # Path to your Legal-BERT model
        groq_api_key=settings.GROQ_API_KEY,  # Optional: for answer generation
        use_colbert=True
    )
    
    # Example legal documents
    documents = [
        {
            'title': 'My Legal Document', # You can provide any title
            'text': "This is a sample legal document text."
        }
    ]
    
    # Create a session
    session_id = rag.create_session("Sample Session")
    
    # Add documents to the session
    rag.add_documents_to_session(documents, session_id)
    
    # Example queries
    queries = [
        "What was the main argument in the case?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        result = rag.query_session(query, session_id, top_k=3)
        
        if 'error' not in result:
            print(f"Query: {query}")
            print(f"Query Type: {result['query_analysis']['type']}")
            print(f"Confidence: {result.get('confidence', 0):.1f}%")
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} (Relevance: {source['relevance_score']:.2f})")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
