"""
Information Extraction Module for DocIntel

Implements multiple approaches for entity extraction:
- Regex-based extraction (dates, emails, metrics, numbers)
- SpaCy NER (Person, Organization, Location, Money)
- Optional: Transformers-based NER
"""

import re
import spacy
from typing import List, Dict, Optional, Tuple, Set
import pandas as pd
from collections import defaultdict, Counter
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """Multi-method entity extraction system"""
    
    def __init__(self, use_spacy: bool = True, use_transformers: bool = False):
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        
        # Initialize spaCy
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("SpaCy NER model loaded successfully")
            except Exception as e:
                logger.warning(f"SpaCy initialization failed: {e}")
                self.use_spacy = False
                self.nlp = None
        
        # Initialize transformers pipeline
        if use_transformers:
            try:
                from transformers import pipeline
                self.ner_pipeline = pipeline("ner", 
                                            model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                            aggregation_strategy="simple")
                logger.info("Transformers NER pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"Transformers initialization failed: {e}")
                self.use_transformers = False
                self.ner_pipeline = None
        
        # Compile regex patterns
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for various entity types"""
        
        # Date patterns
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b'     # DD Month YYYY
        ]
        
        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Phone number patterns
        self.phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',                    # XXX-XXX-XXXX
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',             # (XXX) XXX-XXXX
            r'\b\d{3}\.\d{3}\.\d{4}\b',                  # XXX.XXX.XXXX
            r'\b\d{10}\b'                                # XXXXXXXXXX
        ]
        
        # Money/Currency patterns
        self.money_patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',        # $X,XXX.XX
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|usd)\b',  # X dollars/USD
            r'\b(?:EUR|GBP|JPY)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'       # EUR/GBP/JPY amounts
        ]
        
        # Percentage pattern
        self.percentage_pattern = r'\b\d+(?:\.\d+)?%\b'
        
        # Number patterns (general)
        self.number_patterns = [
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',        # General numbers with commas
            r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand|k|M|B|T)\b',  # Large numbers
        ]
        
        # URL pattern
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        # Compile all patterns
        self.compiled_patterns = {
            'dates': [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns],
            'emails': re.compile(self.email_pattern),
            'phones': [re.compile(pattern) for pattern in self.phone_patterns],
            'money': [re.compile(pattern, re.IGNORECASE) for pattern in self.money_patterns],
            'percentages': re.compile(self.percentage_pattern),
            'numbers': [re.compile(pattern, re.IGNORECASE) for pattern in self.number_patterns],
            'urls': re.compile(self.url_pattern)
        }
    
    def extract_regex_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns"""
        
        entities = defaultdict(list)
        
        # Extract dates
        for pattern in self.compiled_patterns['dates']:
            matches = pattern.findall(text)
            entities['dates'].extend(matches)
        
        # Extract emails
        matches = self.compiled_patterns['emails'].findall(text)
        entities['emails'].extend(matches)
        
        # Extract phone numbers
        for pattern in self.compiled_patterns['phones']:
            matches = pattern.findall(text)
            entities['phones'].extend(matches)
        
        # Extract money amounts
        for pattern in self.compiled_patterns['money']:
            matches = pattern.findall(text)
            entities['money'].extend(matches)
        
        # Extract percentages
        matches = self.compiled_patterns['percentages'].findall(text)
        entities['percentages'].extend(matches)
        
        # Extract numbers
        for pattern in self.compiled_patterns['numbers']:
            matches = pattern.findall(text)
            entities['numbers'].extend(matches)
        
        # Extract URLs
        matches = self.compiled_patterns['urls'].findall(text)
        entities['urls'].extend(matches)
        
        # Remove duplicates and convert to regular dict
        cleaned_entities = {}
        for entity_type, entity_list in entities.items():
            cleaned_entities[entity_type] = list(set(entity_list))
        
        return cleaned_entities
    
    def extract_spacy_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using spaCy NER"""
        
        if not self.use_spacy or not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            entities = defaultdict(list)
            
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'score', 1.0)  # spaCy doesn't provide confidence by default
                }
                entities[ent.label_].append(entity_info)
            
            return dict(entities)
        
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
            return {}
    
    def extract_transformers_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using transformers NER pipeline"""
        
        if not self.use_transformers or not self.ner_pipeline:
            return {}
        
        try:
            # Split text into chunks if too long (transformer models have token limits)
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            all_entities = defaultdict(list)
            
            for chunk_start, chunk in enumerate(chunks):
                entities = self.ner_pipeline(chunk)
                
                for entity in entities:
                    entity_info = {
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'start': entity['start'] + chunk_start * max_length,
                        'end': entity['end'] + chunk_start * max_length,
                        'confidence': entity['score']
                    }
                    all_entities[entity['entity_group']].append(entity_info)
            
            return dict(all_entities)
        
        except Exception as e:
            logger.error(f"Error in transformers entity extraction: {e}")
            return {}
    
    def extract_all_entities(self, text: str) -> Dict[str, Dict]:
        """Extract entities using all available methods"""
        
        results = {}
        
        # Regex extraction
        regex_entities = self.extract_regex_entities(text)
        if regex_entities:
            results['regex'] = regex_entities
        
        # SpaCy extraction
        spacy_entities = self.extract_spacy_entities(text)
        if spacy_entities:
            results['spacy'] = spacy_entities
        
        # Transformers extraction
        if self.use_transformers:
            transformers_entities = self.extract_transformers_entities(text)
            if transformers_entities:
                results['transformers'] = transformers_entities
        
        return results
    
    def extract_document_entities(self, document: Dict, text_key: str = 'text') -> Dict:
        """Extract entities from a single document"""
        
        text = document.get(text_key, '')
        if not text:
            return document
        
        # Extract entities
        entities = self.extract_all_entities(text)
        
        # Add entities to document
        new_doc = document.copy()
        new_doc['entities'] = entities
        new_doc['entity_summary'] = self._summarize_entities(entities)
        
        return new_doc
    
    def extract_corpus_entities(self, documents: List[Dict], text_key: str = 'text') -> List[Dict]:
        """Extract entities from multiple documents"""
        
        logger.info(f"Extracting entities from {len(documents)} documents...")
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self.extract_document_entities(doc, text_key)
                processed_docs.append(processed_doc)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def _summarize_entities(self, entities: Dict[str, Dict]) -> Dict:
        """Create a summary of extracted entities"""
        
        summary = {}
        
        for method, method_entities in entities.items():
            method_summary = {}
            
            for entity_type, entity_list in method_entities.items():
                if isinstance(entity_list, list):
                    if entity_list and isinstance(entity_list[0], dict):
                        # For structured entities (spaCy, transformers)
                        method_summary[entity_type] = {
                            'count': len(entity_list),
                            'examples': [e['text'] for e in entity_list[:3]]  # First 3 examples
                        }
                    else:
                        # For simple entities (regex)
                        method_summary[entity_type] = {
                            'count': len(entity_list),
                            'examples': entity_list[:3]  # First 3 examples
                        }
            
            summary[method] = method_summary
        
        return summary
    
    def get_entity_statistics(self, documents: List[Dict]) -> Dict:
        """Get statistics about entities across the corpus"""
        
        stats = {
            'total_documents': len(documents),
            'documents_with_entities': 0,
            'entity_counts': defaultdict(int),
            'entity_types': defaultdict(set),
            'most_common_entities': {}
        }
        
        all_entities = defaultdict(Counter)
        
        for doc in documents:
            if 'entities' not in doc:
                continue
            
            stats['documents_with_entities'] += 1
            
            for method, method_entities in doc['entities'].items():
                for entity_type, entity_list in method_entities.items():
                    stats['entity_counts'][f"{method}_{entity_type}"] += len(entity_list)
                    
                    # Collect entity texts
                    if isinstance(entity_list, list) and entity_list:
                        if isinstance(entity_list[0], dict):
                            # Structured entities
                            entity_texts = [e['text'] for e in entity_list]
                        else:
                            # Simple entities
                            entity_texts = entity_list
                        
                        all_entities[f"{method}_{entity_type}"].update(entity_texts)
        
        # Get most common entities for each type
        for entity_type, counter in all_entities.items():
            stats['most_common_entities'][entity_type] = counter.most_common(5)
        
        return dict(stats)
    
    def save_entities(self, documents: List[Dict], filename: str = "entities.json"):
        """Save extracted entities to file"""
        
        # Prepare data for JSON serialization
        entities_data = []
        
        for doc in documents:
            if 'entities' in doc:
                doc_data = {
                    'id': doc.get('id', 'unknown'),
                    'title': doc.get('title', 'Untitled'),
                    'entities': doc['entities'],
                    'entity_summary': doc.get('entity_summary', {})
                }
                entities_data.append(doc_data)
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved entities for {len(entities_data)} documents to {filename}")


def main():
    """Demo function to test entity extraction"""
    
    # Initialize extractor
    extractor = EntityExtractor(use_spacy=True, use_transformers=False)
    
    # Sample documents with various entity types
    sample_docs = [
        {
            'id': 'doc1',
            'title': 'Business Meeting',
            'text': '''The quarterly meeting was held on March 15, 2024, with CEO John Smith and CFO Sarah Johnson. 
                      They discussed the $2.5 million investment and the 15% growth target for this year. 
                      Contact them at john.smith@company.com or call (555) 123-4567.'''
        },
        {
            'id': 'doc2',
            'title': 'Market Report',
            'text': '''Apple Inc. reported strong earnings of $89.5 billion for Q1 2024. 
                      The stock price increased by 12.5% following the announcement. 
                      Visit https://investor.apple.com for more details.'''
        },
        {
            'id': 'doc3',
            'title': 'Research Paper',
            'text': '''Dr. Maria Rodriguez from Stanford University published research on 01/20/2024. 
                      The study involved 1,500 participants and showed 78% improvement in outcomes. 
                      Funding of €500,000 was provided by the European Research Council.'''
        }
    ]
    
    # Extract entities
    processed_docs = extractor.extract_corpus_entities(sample_docs)
    
    # Display results
    print("Entity Extraction Results:")
    print("=" * 50)
    
    for doc in processed_docs:
        print(f"\nDocument: {doc['title']}")
        print(f"Text: {doc['text'][:100]}...")
        
        if 'entity_summary' in doc:
            print("Entities found:")
            for method, entities in doc['entity_summary'].items():
                print(f"  {method.upper()}:")
                for entity_type, info in entities.items():
                    print(f"    {entity_type}: {info['count']} ({', '.join(info['examples'])})")
        
        print("-" * 30)
    
    # Get statistics
    stats = extractor.get_entity_statistics(processed_docs)
    print(f"\nEntity Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Documents with entities: {stats['documents_with_entities']}")
    
    print("\nEntity counts:")
    for entity_type, count in stats['entity_counts'].items():
        print(f"  {entity_type}: {count}")
    
    print("\nMost common entities:")
    for entity_type, common_entities in stats['most_common_entities'].items():
        if common_entities:
            print(f"  {entity_type}: {common_entities[:3]}")


if __name__ == "__main__":
    main()
