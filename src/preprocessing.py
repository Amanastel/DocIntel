"""
Text Preprocessing Module for DocIntel

Handles comprehensive text preprocessing including:
- Tokenization
- Lowercasing
- Stopword removal
- Lemmatization
- Punctuation handling
- Text cleaning
"""

import re
import string
import nltk
import spacy
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from collections import Counter
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(self, use_spacy: bool = True, language: str = 'en'):
        self.use_spacy = use_spacy
        self.language = language
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.stem import WordNetLemmatizer
            
            self.nltk_stopwords = set(stopwords.words('english'))
            self.nltk_lemmatizer = WordNetLemmatizer()
            self.word_tokenize = word_tokenize
            self.sent_tokenize = sent_tokenize
            
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
            self.nltk_stopwords = set()
        
        # Initialize spaCy if requested
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("SpaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"SpaCy initialization failed: {e}")
                self.use_spacy = False
                self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_punctuation(self, text: str, keep_sentences: bool = True) -> str:
        """Remove punctuation from text"""
        if keep_sentences:
            # Keep sentence-ending punctuation
            text = re.sub(r'[^\w\s.!?]', '', text)
        else:
            # Remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def tokenize_text(self, text: str, method: str = 'nltk') -> List[str]:
        """Tokenize text into words"""
        if method == 'spacy' and self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space]
        
        elif method == 'nltk' and hasattr(self, 'word_tokenize'):
            return self.word_tokenize(text)
        
        else:
            # Fallback to simple split
            return text.split()
    
    def remove_stopwords(self, tokens: List[str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
        """Remove stopwords from token list"""
        stopwords = self.nltk_stopwords.copy()
        
        if custom_stopwords:
            stopwords.update(custom_stopwords)
        
        return [token for token in tokens if token.lower() not in stopwords]
    
    def lemmatize_tokens(self, tokens: List[str], method: str = 'spacy') -> List[str]:
        """Lemmatize tokens"""
        if method == 'spacy' and self.use_spacy and self.nlp:
            # Process tokens with spaCy
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        
        elif method == 'nltk' and hasattr(self, 'nltk_lemmatizer'):
            return [self.nltk_lemmatizer.lemmatize(token) for token in tokens]
        
        else:
            # Return original tokens if no lemmatization available
            return tokens
    
    def filter_tokens(self, tokens: List[str], min_length: int = 2, 
                     remove_numbers: bool = True, remove_single_chars: bool = True) -> List[str]:
        """Filter tokens based on various criteria"""
        filtered = []
        
        for token in tokens:
            # Skip empty tokens
            if not token:
                continue
            
            # Skip if too short
            if len(token) < min_length:
                continue
            
            # Skip single characters if requested
            if remove_single_chars and len(token) == 1:
                continue
            
            # Skip numbers if requested
            if remove_numbers and token.isdigit():
                continue
            
            # Skip if only punctuation
            if all(c in string.punctuation for c in token):
                continue
            
            filtered.append(token)
        
        return filtered
    
    def preprocess_document(self, text: str, 
                          lowercase: bool = True,
                          remove_punct: bool = True,
                          remove_stops: bool = True,
                          lemmatize: bool = True,
                          min_token_length: int = 2,
                          custom_stopwords: Optional[List[str]] = None) -> Dict:
        """Complete preprocessing pipeline for a single document"""
        
        # Store original text
        original_text = text
        
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Lowercase
        if lowercase:
            text = text.lower()
        
        # Step 3: Remove punctuation
        if remove_punct:
            text = self.remove_punctuation(text, keep_sentences=False)
        
        # Step 4: Tokenize
        tokens = self.tokenize_text(text)
        
        # Step 5: Remove stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens, custom_stopwords)
        
        # Step 6: Lemmatize
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Step 7: Filter tokens
        tokens = self.filter_tokens(tokens, min_length=min_token_length)
        
        # Create result dictionary
        result = {
            'original_text': original_text,
            'cleaned_text': text,
            'tokens': tokens,
            'processed_text': ' '.join(tokens),
            'token_count': len(tokens),
            'original_length': len(original_text),
            'processed_length': len(' '.join(tokens))
        }
        
        return result
    
    def preprocess_documents(self, documents: List[Dict], 
                           text_key: str = 'text',
                           **preprocessing_kwargs) -> List[Dict]:
        """Preprocess multiple documents"""
        
        logger.info(f"Preprocessing {len(documents)} documents...")
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                # Get text from document
                text = doc.get(text_key, '')
                
                # Preprocess the text
                processed = self.preprocess_document(text, **preprocessing_kwargs)
                
                # Create new document with processed data
                new_doc = doc.copy()
                new_doc.update(processed)
                
                processed_docs.append(new_doc)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error preprocessing document {i}: {e}")
                continue
        
        logger.info(f"Successfully preprocessed {len(processed_docs)} documents")
        return processed_docs
    
    def get_vocabulary(self, documents: List[Dict], 
                      text_key: str = 'processed_text',
                      min_freq: int = 1) -> Dict[str, int]:
        """Build vocabulary from processed documents"""
        
        word_counts = Counter()
        
        for doc in documents:
            if text_key in doc:
                words = doc[text_key].split()
                word_counts.update(words)
        
        # Filter by minimum frequency
        vocabulary = {word: count for word, count in word_counts.items() 
                     if count >= min_freq}
        
        return vocabulary
    
    def get_preprocessing_stats(self, documents: List[Dict]) -> Dict:
        """Get statistics about preprocessing results"""
        
        if not documents:
            return {}
        
        original_lengths = [doc.get('original_length', 0) for doc in documents]
        processed_lengths = [doc.get('processed_length', 0) for doc in documents]
        token_counts = [doc.get('token_count', 0) for doc in documents]
        
        stats = {
            'total_documents': len(documents),
            'avg_original_length': np.mean(original_lengths),
            'avg_processed_length': np.mean(processed_lengths),
            'avg_token_count': np.mean(token_counts),
            'total_tokens': sum(token_counts),
            'compression_ratio': np.mean(processed_lengths) / np.mean(original_lengths) if np.mean(original_lengths) > 0 else 0
        }
        
        return stats


def main():
    """Demo function to test preprocessing"""
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_spacy=True)
    
    # Sample documents
    sample_docs = [
        {
            'id': 'doc1',
            'text': 'The quick brown fox jumps over the lazy dog! This is a test document with punctuation, numbers like 123, and various other elements.',
            'title': 'Sample Document 1'
        },
        {
            'id': 'doc2', 
            'text': 'Natural Language Processing (NLP) is a fascinating field that combines computer science, artificial intelligence, and linguistics.',
            'title': 'Sample Document 2'
        },
        {
            'id': 'doc3',
            'text': 'Machine learning algorithms can process large amounts of data to identify patterns and make predictions.',
            'title': 'Sample Document 3'
        }
    ]
    
    # Preprocess documents
    processed_docs = preprocessor.preprocess_documents(sample_docs)
    
    # Display results
    print("Preprocessing Results:")
    print("=" * 50)
    
    for doc in processed_docs:
        print(f"\nDocument: {doc['title']}")
        print(f"Original: {doc['original_text']}")
        print(f"Processed: {doc['processed_text']}")
        print(f"Tokens: {doc['tokens']}")
        print(f"Token count: {doc['token_count']}")
        print("-" * 30)
    
    # Get vocabulary
    vocab = preprocessor.get_vocabulary(processed_docs)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Most common words: {sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    # Get stats
    stats = preprocessor.get_preprocessing_stats(processed_docs)
    print(f"\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
