"""
Data Loader Module for DocIntel

Supports multiple data sources:
- NLTK Reuters corpus
- Custom datasets
- Kaggle datasets
- ArXiv abstracts
"""

import os
import nltk
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handle loading and managing various document datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.documents = []
        self.metadata = {}
        
    def load_reuters_corpus(self, max_docs: Optional[int] = None) -> List[Dict]:
        """Load NLTK Reuters corpus"""
        try:
            nltk.download('reuters', quiet=True)
            from nltk.corpus import reuters
            
            logger.info("Loading Reuters corpus...")
            
            # Get all file IDs
            fileids = reuters.fileids()
            if max_docs:
                fileids = fileids[:max_docs]
            
            documents = []
            for fileid in fileids:
                try:
                    doc = {
                        'id': fileid,
                        'text': reuters.raw(fileid),
                        'categories': reuters.categories(fileid),
                        'source': 'reuters',
                        'title': f"Reuters_{fileid}",
                        'length': len(reuters.raw(fileid))
                    }
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error loading document {fileid}: {e}")
                    continue
            
            self.documents = documents
            self.metadata = {
                'source': 'reuters',
                'total_docs': len(documents),
                'categories': list(set([cat for doc in documents for cat in doc['categories']]))
            }
            
            logger.info(f"Loaded {len(documents)} Reuters documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Reuters corpus: {e}")
            return []
    
    def load_custom_dataset(self, file_path: str, text_column: str = 'text', 
                          title_column: str = 'title', format: str = 'csv') -> List[Dict]:
        """Load custom dataset from CSV/JSON file"""
        try:
            file_path = Path(file_path)
            
            if format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif format.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            documents = []
            for idx, row in df.iterrows():
                doc = {
                    'id': f"custom_{idx}",
                    'text': str(row[text_column]),
                    'title': str(row.get(title_column, f"Document_{idx}")),
                    'source': 'custom',
                    'length': len(str(row[text_column]))
                }
                
                # Add other columns as metadata
                for col in df.columns:
                    if col not in [text_column, title_column]:
                        doc[col] = row[col]
                
                documents.append(doc)
            
            self.documents = documents
            self.metadata = {
                'source': 'custom',
                'total_docs': len(documents),
                'file_path': str(file_path)
            }
            
            logger.info(f"Loaded {len(documents)} custom documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            return []
    
    def create_sample_dataset(self, num_docs: int = 100) -> List[Dict]:
        """Create a sample dataset for testing"""
        sample_texts = [
            "The stock market experienced significant volatility today as investors reacted to the latest economic indicators.",
            "A new breakthrough in artificial intelligence has been announced by researchers at leading universities.",
            "Climate change continues to be a major concern for policymakers around the world.",
            "The technology sector shows strong growth potential despite recent market uncertainties.",
            "Healthcare innovations are transforming patient care and medical outcomes globally.",
            "Financial markets are closely watching central bank policy decisions.",
            "Renewable energy investments are increasing as countries pursue sustainability goals.",
            "Supply chain disruptions continue to impact global trade and commerce.",
            "Educational institutions are adapting to new digital learning methodologies.",
            "Consumer behavior patterns are shifting due to technological advancements."
        ]
        
        categories = ['finance', 'technology', 'healthcare', 'environment', 'education']
        
        documents = []
        for i in range(num_docs):
            base_text = np.random.choice(sample_texts)
            doc = {
                'id': f"sample_{i}",
                'text': base_text + f" Document {i} provides additional context and analysis.",
                'title': f"Sample Document {i}",
                'categories': [np.random.choice(categories)],
                'source': 'sample',
                'length': len(base_text) + 50
            }
            documents.append(doc)
        
        self.documents = documents
        self.metadata = {
            'source': 'sample',
            'total_docs': len(documents),
            'categories': categories
        }
        
        logger.info(f"Created {len(documents)} sample documents")
        return documents
    
    def get_document_stats(self) -> Dict:
        """Get statistics about loaded documents"""
        if not self.documents:
            return {}
        
        lengths = [doc['length'] for doc in self.documents]
        
        stats = {
            'total_documents': len(self.documents),
            'avg_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'std_length': np.std(lengths)
        }
        
        # Add category information if available
        if 'categories' in self.metadata:
            stats['total_categories'] = len(self.metadata['categories'])
            stats['categories'] = self.metadata['categories']
        
        return stats
    
    def save_documents(self, filename: str = "documents.json"):
        """Save loaded documents to file"""
        filepath = self.data_dir / filename
        
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'stats': self.get_document_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.documents)} documents to {filepath}")
    
    def load_documents(self, filename: str = "documents.json") -> List[Dict]:
        """Load documents from saved file"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data.get('documents', [])
            self.metadata = data.get('metadata', {})
            
            logger.info(f"Loaded {len(self.documents)} documents from {filepath}")
            return self.documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Filter documents by category"""
        filtered_docs = []
        for doc in self.documents:
            if 'categories' in doc and category in doc['categories']:
                filtered_docs.append(doc)
        return filtered_docs
    
    def get_sample_documents(self, n: int = 5) -> List[Dict]:
        """Get a random sample of documents"""
        if not self.documents:
            return []
        
        sample_size = min(n, len(self.documents))
        return np.random.choice(self.documents, size=sample_size, replace=False).tolist()


def main():
    """Demo function to test the data loader"""
    loader = DataLoader()
    
    # Try to load Reuters corpus first
    print("Loading Reuters corpus...")
    docs = loader.load_reuters_corpus(max_docs=100)
    
    if not docs:
        print("Reuters corpus not available, creating sample dataset...")
        docs = loader.create_sample_dataset(50)
    
    # Print statistics
    stats = loader.get_document_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample documents
    print("\nSample Documents:")
    samples = loader.get_sample_documents(3)
    for i, doc in enumerate(samples, 1):
        print(f"\n{i}. {doc['title']}")
        print(f"   Categories: {doc.get('categories', 'N/A')}")
        print(f"   Length: {doc['length']} characters")
        print(f"   Preview: {doc['text'][:100]}...")
    
    # Save documents
    loader.save_documents()
    print(f"\nSaved documents to data/documents.json")


if __name__ == "__main__":
    main()
