import spacy
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LegalNERProcessor:
    """
    Wrapper for the spaCy NER model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the NER processor
        
        Args:
            model_path: Path to the trained spaCy model
        """
        try:
            self.nlp = spacy.load(model_path)
            self.nlp.max_length = 5000000  # 5M characters instead of default 1M
            logger.info(f"Loaded spaCy model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            # Fallback to a basic English model if available
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp.max_length = 5000000
                logger.warning("Using fallback English model")
            except:
                raise Exception("No spaCy model available")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text and extract named entities
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with NER results
        """
        try:
            if len(text) > 4000000:  # If text is larger than 4M chars, chunk it
                logger.info(f"Text too large ({len(text)} chars), processing in chunks")
                return self._process_large_text(text)
            
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            entity_counts = {}
            
            for ent in doc.ents:
                processed_entities = self._process_entity(ent)
                
                for entity_text, entity_label in processed_entities:
                    entity_info = {
                        "text": entity_text,
                        "label": entity_label,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    entities.append(entity_info)
                    
                    # Count entities by type
                    if entity_label not in entity_counts:
                        entity_counts[entity_label] = []
                    entity_counts[entity_label].append(entity_text)
            
            # Remove duplicates and count
            for label in entity_counts:
                unique_entities = list(set(entity_counts[label]))
                entity_counts[label] = {
                    "entities": unique_entities,
                    "count": len(unique_entities)
                }
            
            return {
                "entities": entities,
                "entity_counts": entity_counts,
                "total_entities": len(entities),
                "unique_labels": list(entity_counts.keys())
            }
            
        except Exception as e:
            logger.error(f"Error processing text with NER: {str(e)}")
            return {
                "error": str(e),
                "entities": [],
                "entity_counts": {},
                "total_entities": 0
            }
    
    def _process_large_text(self, text: str, chunk_size: int = 3000000) -> Dict[str, Any]:
        """
        Process very large texts by chunking
        """
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_entities = []
        all_entity_counts = {}
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                doc = self.nlp(chunk)
                
                for ent in doc.ents:
                    processed_entities = self._process_entity(ent)
                    
                    for entity_text, entity_label in processed_entities:
                        entity_info = {
                            "text": entity_text,
                            "label": entity_label,
                            "start": ent.start_char + (i * chunk_size),  # Adjust position
                            "end": ent.end_char + (i * chunk_size)
                        }
                        all_entities.append(entity_info)
                        
                        # Count entities by type
                        if entity_label not in all_entity_counts:
                            all_entity_counts[entity_label] = []
                        all_entity_counts[entity_label].append(entity_text)
                        
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Remove duplicates and count
        for label in all_entity_counts:
            unique_entities = list(set(all_entity_counts[label]))
            all_entity_counts[label] = {
                "entities": unique_entities,
                "count": len(unique_entities)
            }
        
        return {
            "entities": all_entities,
            "entity_counts": all_entity_counts,
            "total_entities": len(all_entities),
            "unique_labels": list(all_entity_counts.keys()),
            "processed_in_chunks": True,
            "num_chunks": len(chunks)
        }
    
    def _process_entity(self, ent) -> List[tuple]:
        """
        Clean and split merged entities - exact implementation from user's code
        """
        # Split ORG/PRECEDENT entities joined with " and "
        if ent.label_ in ["PRECEDENT", "ORG"] and " and " in ent.text:
            parts = ent.text.split(" and ")
            return [(p.strip(), "ORG") for p in parts]
        return [(ent.text, ent.label_)]
