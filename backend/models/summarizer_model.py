import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Tuple, Dict
import re
from dataclasses import dataclass
from sentence_transformers import util
import networkx as nx
from groq import Groq

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@dataclass
class SentenceScore:
    """Store sentence information with multiple scoring metrics"""
    text: str
    index: int
    position_score: float
    embedding: np.ndarray
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    legal_term_score: float = 0.0
    combined_score: float = 0.0

class LegalDocumentSummarizer:
    def __init__(self, model_path: str, groq_api_key: str = None):
        """
        Initialize the Legal Document Summarizer
        
        Args:
            model_path: Path to Legal-BERT model
            groq_api_key: API key for Groq (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load Legal-BERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Groq client if API key provided
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        
        # Legal terms for domain-specific scoring
        self.legal_terms = {
            'high_importance': ['verdict', 'judgment', 'ruling', 'held', 'ordered', 
                              'convicted', 'acquitted', 'liable', 'damages', 'injunction'],
            'medium_importance': ['plaintiff', 'defendant', 'court', 'appeal', 'evidence',
                                'testimony', 'witness', 'statute', 'precedent', 'jurisdiction'],
            'low_importance': ['whereas', 'pursuant', 'herein', 'thereof', 'hereby']
        }
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess and segment text into sentences
        
        Args:
            text: Raw legal document text
            
        Returns:
            List of cleaned sentences
        """
        # Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        return sentences
    
    def get_sentence_embeddings(self, sentences: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for sentences using Legal-BERT
        
        Args:
            sentences: List of sentences
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of sentence embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Use mean pooling for sentence representation
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9
                )
                
                embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def calculate_position_scores(self, num_sentences: int) -> List[float]:
        """
        Calculate position-based scores (higher for beginning and end)
        
        Args:
            num_sentences: Total number of sentences
            
        Returns:
            List of position scores
        """
        scores = []
        for i in range(num_sentences):
            if i < 3:  # First 3 sentences
                score = 1.0 - (i * 0.1)
            elif i >= num_sentences - 3:  # Last 3 sentences
                score = 0.8 - ((num_sentences - i - 1) * 0.1)
            else:  # Middle sentences
                score = 0.5
            scores.append(score)
        return scores
    
    def calculate_legal_term_scores(self, sentences: List[str]) -> List[float]:
        """
        Score sentences based on presence of legal terms
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of legal term scores
        """
        scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0.0
            
            for term in self.legal_terms['high_importance']:
                if term in sentence_lower:
                    score += 3.0
            
            for term in self.legal_terms['medium_importance']:
                if term in sentence_lower:
                    score += 2.0
            
            for term in self.legal_terms['low_importance']:
                if term in sentence_lower:
                    score += 1.0
            
            # Normalize by sentence length
            score = score / (len(sentence.split()) ** 0.5)
            scores.append(score)
        
        return scores
    
    def calculate_tfidf_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate TF-IDF based importance scores
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of TF-IDF scores
        """
        if len(sentences) < 2:
            return [1.0] * len(sentences)
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # Sum TF-IDF values for each sentence
            scores = tfidf_matrix.sum(axis=1).A1
            # Normalize scores
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores.tolist()
        except:
            return [1.0] * len(sentences)
    
    def textrank_scoring(self, embeddings: np.ndarray, damping: float = 0.85) -> List[float]:
        """
        Apply TextRank algorithm for sentence importance
        
        Args:
            embeddings: Sentence embeddings
            damping: Damping factor for PageRank
            
        Returns:
            List of TextRank scores
        """
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank
        try:
            scores = nx.pagerank(graph, alpha=damping)
            return [scores[i] for i in range(len(scores))]
        except:
            return [1.0] * len(embeddings)
    
    def mmr_selection(self, sentence_scores: List[SentenceScore], 
                     num_sentences: int, lambda_param: float = 0.7) -> List[SentenceScore]:
        """
        Maximal Marginal Relevance for diverse sentence selection
        
        Args:
            sentence_scores: List of scored sentences
            num_sentences: Number of sentences to select
            lambda_param: Trade-off between relevance and diversity
            
        Returns:
            Selected sentences
        """
        if len(sentence_scores) <= num_sentences:
            return sentence_scores
        
        selected = []
        remaining = sentence_scores.copy()
        
        # Select first sentence with highest score
        first = max(remaining, key=lambda x: x.combined_score)
        selected.append(first)
        remaining.remove(first)
        
        # Select remaining sentences using MMR
        while len(selected) < num_sentences and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score
                relevance = candidate.combined_score
                
                # Calculate maximum similarity to selected sentences
                max_sim = max([
                    cosine_similarity(
                        candidate.embedding.reshape(1, -1),
                        s.embedding.reshape(1, -1)
                    )[0, 0]
                    for s in selected
                ])
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # Select sentence with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected
    
    def extract_summary(self, text: str, num_sentences: int = 15, 
                       use_mmr: bool = True, lambda_mmr: float = 0.7) -> Tuple[List[str], List[int]]:
        """
        Main extraction method
        
        Args:
            text: Legal document text
            num_sentences: Number of sentences to extract
            use_mmr: Whether to use MMR for diversity
            lambda_mmr: MMR parameter for relevance vs diversity trade-off
            
        Returns:
            Tuple of (selected sentences, original indices)
        """
        # Preprocess text
        sentences = self.preprocess_text(text)
        if not sentences:
            return [], []
        
        print(f"Processing {len(sentences)} sentences...")
        
        # Get embeddings
        embeddings = self.get_sentence_embeddings(sentences)
        
        # Calculate various scores
        position_scores = self.calculate_position_scores(len(sentences))
        tfidf_scores = self.calculate_tfidf_scores(sentences)
        legal_scores = self.calculate_legal_term_scores(sentences)
        textrank_scores = self.textrank_scoring(embeddings)
        
        # Calculate document centroid
        doc_centroid = np.mean(embeddings, axis=0)
        semantic_scores = [
            cosine_similarity(
                emb.reshape(1, -1), 
                doc_centroid.reshape(1, -1)
            )[0, 0]
            for emb in embeddings
        ]
        
        # Create SentenceScore objects
        sentence_scores = []
        for i, sent in enumerate(sentences):
            score_obj = SentenceScore(
                text=sent,
                index=i,
                position_score=position_scores[i],
                embedding=embeddings[i],
                tfidf_score=tfidf_scores[i],
                semantic_score=semantic_scores[i],
                legal_term_score=legal_scores[i]
            )
            
            # Combine scores with weights
            score_obj.combined_score = (
                0.15 * score_obj.position_score +
                0.25 * score_obj.tfidf_score +
                0.35 * score_obj.semantic_score +
                0.25 * score_obj.legal_term_score
            )
            
            sentence_scores.append(score_obj)
        
        # Select sentences
        if use_mmr:
            selected_scores = self.mmr_selection(sentence_scores, num_sentences, lambda_mmr)
        else:
            # Simple top-k selection
            selected_scores = sorted(sentence_scores, 
                                    key=lambda x: x.combined_score, 
                                    reverse=True)[:num_sentences]
        
        # Sort by original order
        selected_scores = sorted(selected_scores, key=lambda x: x.index)
        
        # Extract sentences and indices
        selected_sentences = [s.text for s in selected_scores]
        selected_indices = [s.index for s in selected_scores]
        
        return selected_sentences, selected_indices
    
    def refine_with_groq(self, sentences: List[str]) -> str:
        """
        Use Groq API to improve flow and readability
        
        Args:
            sentences: List of extracted sentences
            
        Returns:
            Refined summary
        """
        if not self.groq_client:
            return ' '.join(sentences)
        
        try:
            prompt = f"""You are a legal document editor. The following sentences were extracted from a legal case. 
            Please improve the flow and readability while maintaining all the legal facts and information exactly as stated.
            Only make minor adjustments for better flow - do not add or remove any substantive information.
            
            Extracted sentences:
            {' '.join(sentences)}
            
            Refined summary:"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a legal document editor focused on clarity and precision."},
                          {"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Or another available model
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return ' '.join(sentences)
    
    def summarize(self, text: str, num_sentences: int = 15, 
                 use_groq: bool = True, use_mmr: bool = True) -> Dict:
        """
        Complete summarization pipeline
        
        Args:
            text: Legal document text
            num_sentences: Number of sentences to extract
            use_groq: Whether to use Groq for refinement
            use_mmr: Whether to use MMR for diversity
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            print(f"[v0] Starting summarization for text of length {len(text)}")
            
            # Extract sentences
            print(f"[v0] Extracting {num_sentences} sentences...")
            extracted_sentences, indices = self.extract_summary(
                text, num_sentences, use_mmr
            )
            
            if not extracted_sentences:
                print("[v0] No sentences extracted")
                return {
                    'summary': '',
                    'extractive_summary': '',
                    'extracted_sentences': [],
                    'sentence_indices': [],
                    'refined': False,
                    'error': 'No sentences could be extracted from the text'
                }
            
            print(f"[v0] Extracted {len(extracted_sentences)} sentences")
            
            # Create extractive summary
            extractive_summary = ' '.join(extracted_sentences)
            print(f"[v0] Created extractive summary of length {len(extractive_summary)}")
            
            # Refine with Groq if requested
            if use_groq and self.groq_client:
                print("[v0] Refining with Groq...")
                try:
                    refined_summary = self.refine_with_groq(extracted_sentences)
                    print("[v0] Groq refinement completed")
                    return {
                        'summary': refined_summary,
                        'extractive_summary': extractive_summary,
                        'extracted_sentences': extracted_sentences,
                        'sentence_indices': indices,
                        'refined': True
                    }
                except Exception as e:
                    print(f"[v0] Groq refinement failed: {e}")
                    # Fall back to extractive summary
                    return {
                        'summary': extractive_summary,
                        'extractive_summary': extractive_summary,
                        'extracted_sentences': extracted_sentences,
                        'sentence_indices': indices,
                        'refined': False,
                        'groq_error': str(e)
                    }
            else:
                print("[v0] Using extractive summary (no Groq)")
                return {
                    'summary': extractive_summary,
                    'extracted_sentences': extracted_sentences,
                    'sentence_indices': indices,
                    'refined': False
                }
                
        except Exception as e:
            print(f"[v0] Summarization error: {e}")
            return {
                'summary': '',
                'extractive_summary': '',
                'extracted_sentences': [],
                'sentence_indices': [],
                'refined': False,
                'error': str(e)
            }


# Example usage
if __name__ == "__main__":
    from backend.config import settings
    # Initialize summarizer using environment variables
    model_path = settings.LEGAL_BERT_PATH
    groq_api_key = settings.GROQ_API_KEY

    summarizer = LegalDocumentSummarizer(model_path, groq_api_key)

    # Example legal text
    legal_text = "Your legal text here"

    # Generate summary
    result = summarizer.summarize(
        legal_text,
        num_sentences=5,
        use_groq=True,
        use_mmr=True
    )

    print("Summary:")
    print(result['summary'])
    print(f"\nExtracted {len(result['extracted_sentences'])} sentences")
    print(f"Refined with Groq: {result['refined']}")
