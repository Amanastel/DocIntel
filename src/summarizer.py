"""
Text Summarization Module for DocIntel

Implements multiple summarization approaches:
- Extractive: TextRank, TF-IDF sentence scoring
- Abstractive: T5, BART (via Transformers)
- Evaluation: ROUGE scores
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import logging
from pathlib import Path
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """Multi-method text summarization system"""
    
    def __init__(self, use_transformers: bool = False):
        self.use_transformers = use_transformers
        
        # Initialize transformers models if requested
        if use_transformers:
            try:
                from transformers import pipeline
                self.t5_summarizer = pipeline("summarization", model="t5-small")
                self.bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                logger.info("Transformers summarization models loaded successfully")
            except Exception as e:
                logger.warning(f"Transformers initialization failed: {e}")
                self.use_transformers = False
                self.t5_summarizer = None
                self.bart_summarizer = None
        
        # Initialize sentence tokenizer
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
        except Exception as e:
            logger.warning(f"NLTK sentence tokenizer not available: {e}")
            self.sent_tokenize = self._simple_sent_tokenize
    
    def _simple_sent_tokenize(self, text: str) -> List[str]:
        """Simple sentence tokenization fallback"""
        # Split on sentence endings followed by whitespace and capital letter
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_word_frequencies(self, text: str) -> Dict[str, float]:
        """Compute word frequencies for TF-IDF calculation"""
        # Simple tokenization and cleaning
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        normalized_freq = {word: freq/max_freq for word, freq in word_freq.items()}
        
        return normalized_freq
    
    def _score_sentences_tfidf(self, sentences: List[str], word_frequencies: Dict[str, float]) -> List[float]:
        """Score sentences using TF-IDF approach"""
        sentence_scores = []
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = 0.0
            
            if words:
                for word in words:
                    if word in word_frequencies:
                        score += word_frequencies[word]
                score = score / len(words)  # Average score
            
            sentence_scores.append(score)
        
        return sentence_scores
    
    def _create_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Create sentence similarity matrix for TextRank"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
        
        return similarity_matrix
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences using word overlap"""
        words1 = set(re.findall(r'\b\w+\b', sent1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sent2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _textrank_algorithm(self, similarity_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """Implement TextRank algorithm"""
        n = similarity_matrix.shape[0]
        scores = np.ones(n) / n
        
        for _ in range(max_iter):
            new_scores = np.ones(n) * (1 - damping) / n
            
            for i in range(n):
                for j in range(n):
                    if similarity_matrix[j][i] > 0:
                        new_scores[i] += damping * scores[j] * similarity_matrix[j][i] / np.sum(similarity_matrix[j])
            
            # Check for convergence
            if np.allclose(scores, new_scores, atol=1e-6):
                break
                
            scores = new_scores
        
        return scores
    
    def extractive_summarize_tfidf(self, text: str, num_sentences: int = 3) -> Dict:
        """Extractive summarization using TF-IDF sentence scoring"""
        
        # Tokenize into sentences
        sentences = self.sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': text,
                'method': 'tfidf',
                'sentences_used': sentences,
                'compression_ratio': 1.0
            }
        
        # Compute word frequencies
        word_frequencies = self._compute_word_frequencies(text)
        
        # Score sentences
        sentence_scores = self._score_sentences_tfidf(sentences, word_frequencies)
        
        # Select top sentences
        sentence_indices = sorted(range(len(sentence_scores)), 
                                key=lambda i: sentence_scores[i], 
                                reverse=True)[:num_sentences]
        
        # Sort selected sentences by original order
        sentence_indices.sort()
        
        selected_sentences = [sentences[i] for i in sentence_indices]
        summary = ' '.join(selected_sentences)
        
        return {
            'summary': summary,
            'method': 'tfidf',
            'sentences_used': selected_sentences,
            'sentence_scores': sentence_scores,
            'compression_ratio': len(summary) / len(text)
        }
    
    def extractive_summarize_textrank(self, text: str, num_sentences: int = 3) -> Dict:
        """Extractive summarization using TextRank algorithm"""
        
        # Tokenize into sentences
        sentences = self.sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': text,
                'method': 'textrank',
                'sentences_used': sentences,
                'compression_ratio': 1.0
            }
        
        # Create similarity matrix
        similarity_matrix = self._create_similarity_matrix(sentences)
        
        # Apply TextRank
        sentence_scores = self._textrank_algorithm(similarity_matrix)
        
        # Select top sentences
        sentence_indices = sorted(range(len(sentence_scores)), 
                                key=lambda i: sentence_scores[i], 
                                reverse=True)[:num_sentences]
        
        # Sort selected sentences by original order
        sentence_indices.sort()
        
        selected_sentences = [sentences[i] for i in sentence_indices]
        summary = ' '.join(selected_sentences)
        
        return {
            'summary': summary,
            'method': 'textrank',
            'sentences_used': selected_sentences,
            'sentence_scores': sentence_scores.tolist(),
            'compression_ratio': len(summary) / len(text)
        }
    
    def abstractive_summarize_t5(self, text: str, max_length: int = 150, min_length: int = 30) -> Dict:
        """Abstractive summarization using T5 model"""
        
        if not self.use_transformers or not self.t5_summarizer:
            return {'error': 'T5 model not available'}
        
        try:
            # T5 expects the input to be prefixed with "summarize: "
            input_text = f"summarize: {text}"
            
            # Generate summary
            result = self.t5_summarizer(input_text, 
                                      max_length=max_length, 
                                      min_length=min_length, 
                                      do_sample=False)
            
            summary = result[0]['summary_text']
            
            return {
                'summary': summary,
                'method': 't5',
                'model': 't5-small',
                'compression_ratio': len(summary) / len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in T5 summarization: {e}")
            return {'error': str(e)}
    
    def abstractive_summarize_bart(self, text: str, max_length: int = 150, min_length: int = 30) -> Dict:
        """Abstractive summarization using BART model"""
        
        if not self.use_transformers or not self.bart_summarizer:
            return {'error': 'BART model not available'}
        
        try:
            # BART can handle the text directly
            result = self.bart_summarizer(text, 
                                        max_length=max_length, 
                                        min_length=min_length, 
                                        do_sample=False)
            
            summary = result[0]['summary_text']
            
            return {
                'summary': summary,
                'method': 'bart',
                'model': 'facebook/bart-large-cnn',
                'compression_ratio': len(summary) / len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in BART summarization: {e}")
            return {'error': str(e)}
    
    def summarize_document(self, document: Dict, 
                          text_key: str = 'text',
                          methods: List[str] = ['tfidf', 'textrank'],
                          num_sentences: int = 3,
                          max_length: int = 150) -> Dict:
        """Summarize a single document using multiple methods"""
        
        text = document.get(text_key, '')
        if not text:
            return document
        
        summaries = {}
        
        # Extractive methods
        if 'tfidf' in methods:
            summaries['tfidf'] = self.extractive_summarize_tfidf(text, num_sentences)
        
        if 'textrank' in methods:
            summaries['textrank'] = self.extractive_summarize_textrank(text, num_sentences)
        
        # Abstractive methods (if available)
        if 't5' in methods and self.use_transformers:
            summaries['t5'] = self.abstractive_summarize_t5(text, max_length)
        
        if 'bart' in methods and self.use_transformers:
            summaries['bart'] = self.abstractive_summarize_bart(text, max_length)
        
        # Add summaries to document
        new_doc = document.copy()
        new_doc['summaries'] = summaries
        
        return new_doc
    
    def summarize_corpus(self, documents: List[Dict], 
                        text_key: str = 'text',
                        methods: List[str] = ['tfidf', 'textrank'],
                        **kwargs) -> List[Dict]:
        """Summarize multiple documents"""
        
        logger.info(f"Summarizing {len(documents)} documents using methods: {methods}")
        
        summarized_docs = []
        
        for i, doc in enumerate(documents):
            try:
                summarized_doc = self.summarize_document(doc, text_key, methods, **kwargs)
                summarized_docs.append(summarized_doc)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Summarized {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error summarizing document {i}: {e}")
                continue
        
        logger.info(f"Successfully summarized {len(summarized_docs)} documents")
        return summarized_docs
    
    def evaluate_summary_rouge(self, reference: str, candidate: str) -> Dict:
        """Evaluate summary using ROUGE metrics (simplified implementation)"""
        
        def get_ngrams(text: str, n: int) -> List[Tuple]:
            words = text.lower().split()
            return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        
        def rouge_n(ref_grams: List, cand_grams: List) -> Dict:
            if not ref_grams:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            ref_counter = Counter(ref_grams)
            cand_counter = Counter(cand_grams)
            
            overlap = sum((ref_counter & cand_counter).values())
            
            precision = overlap / len(cand_grams) if cand_grams else 0.0
            recall = overlap / len(ref_grams) if ref_grams else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {'precision': precision, 'recall': recall, 'f1': f1}
        
        # Calculate ROUGE-1 and ROUGE-2
        ref_1grams = get_ngrams(reference, 1)
        cand_1grams = get_ngrams(candidate, 1)
        rouge1 = rouge_n(ref_1grams, cand_1grams)
        
        ref_2grams = get_ngrams(reference, 2)
        cand_2grams = get_ngrams(candidate, 2)
        rouge2 = rouge_n(ref_2grams, cand_2grams)
        
        return {
            'rouge-1': rouge1,
            'rouge-2': rouge2
        }
    
    def get_summarization_stats(self, documents: List[Dict]) -> Dict:
        """Get statistics about summarization results"""
        
        stats = {
            'total_documents': len(documents),
            'documents_with_summaries': 0,
            'methods_used': set(),
            'avg_compression_ratios': {},
            'summary_lengths': defaultdict(list)
        }
        
        for doc in documents:
            if 'summaries' not in doc:
                continue
            
            stats['documents_with_summaries'] += 1
            
            for method, summary_data in doc['summaries'].items():
                if isinstance(summary_data, dict) and 'summary' in summary_data:
                    stats['methods_used'].add(method)
                    
                    # Track compression ratios
                    if 'compression_ratio' in summary_data:
                        if method not in stats['avg_compression_ratios']:
                            stats['avg_compression_ratios'][method] = []
                        stats['avg_compression_ratios'][method].append(summary_data['compression_ratio'])
                    
                    # Track summary lengths
                    summary_length = len(summary_data['summary'])
                    stats['summary_lengths'][method].append(summary_length)
        
        # Calculate averages
        for method in stats['avg_compression_ratios']:
            ratios = stats['avg_compression_ratios'][method]
            stats['avg_compression_ratios'][method] = sum(ratios) / len(ratios) if ratios else 0.0
        
        stats['methods_used'] = list(stats['methods_used'])
        
        return stats
    
    def save_summaries(self, documents: List[Dict], filename: str = "summaries.json"):
        """Save summaries to file"""
        
        summaries_data = []
        
        for doc in documents:
            if 'summaries' in doc:
                doc_data = {
                    'id': doc.get('id', 'unknown'),
                    'title': doc.get('title', 'Untitled'),
                    'original_length': len(doc.get('text', '')),
                    'summaries': doc['summaries']
                }
                summaries_data.append(doc_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summaries for {len(summaries_data)} documents to {filename}")


def main():
    """Demo function to test summarization"""
    
    # Initialize summarizer
    summarizer = TextSummarizer(use_transformers=False)  # Set to True to test transformers
    
    # Sample documents
    sample_docs = [
        {
            'id': 'doc1',
            'title': 'AI Research Paper',
            'text': '''Artificial intelligence has made significant advances in recent years. 
                      Machine learning algorithms can now process vast amounts of data to identify patterns. 
                      Deep learning models have shown remarkable performance in image recognition tasks. 
                      Natural language processing has enabled computers to understand human language better. 
                      These developments have applications in healthcare, finance, and many other fields. 
                      However, there are still challenges related to bias, interpretability, and ethical considerations. 
                      Future research should focus on making AI systems more transparent and fair.'''
        },
        {
            'id': 'doc2',
            'title': 'Climate Change Report',
            'text': '''Climate change is one of the most pressing issues of our time. 
                      Global temperatures have risen significantly over the past century. 
                      Human activities, particularly fossil fuel consumption, are the primary cause. 
                      The effects include rising sea levels, extreme weather events, and ecosystem disruption. 
                      Immediate action is needed to reduce greenhouse gas emissions. 
                      Renewable energy sources like solar and wind power offer promising solutions. 
                      International cooperation is essential to address this global challenge effectively.'''
        }
    ]
    
    # Summarize documents
    summarized_docs = summarizer.summarize_corpus(sample_docs, methods=['tfidf', 'textrank'])
    
    # Display results
    print("Summarization Results:")
    print("=" * 50)
    
    for doc in summarized_docs:
        print(f"\nDocument: {doc['title']}")
        print(f"Original text ({len(doc['text'])} chars): {doc['text'][:100]}...")
        
        if 'summaries' in doc:
            for method, summary_data in doc['summaries'].items():
                if 'summary' in summary_data:
                    print(f"\n{method.upper()} Summary ({len(summary_data['summary'])} chars):")
                    print(f"  {summary_data['summary']}")
                    print(f"  Compression ratio: {summary_data.get('compression_ratio', 'N/A'):.3f}")
        
        print("-" * 30)
    
    # Get statistics
    stats = summarizer.get_summarization_stats(summarized_docs)
    print(f"\nSummarization Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Documents with summaries: {stats['documents_with_summaries']}")
    print(f"Methods used: {', '.join(stats['methods_used'])}")
    
    print(f"\nAverage compression ratios:")
    for method, ratio in stats['avg_compression_ratios'].items():
        print(f"  {method}: {ratio:.3f}")


if __name__ == "__main__":
    main()
