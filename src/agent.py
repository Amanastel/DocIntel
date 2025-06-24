"""
Agentic Design Module for DocIntel

Implements a research agent that can:
- Accept user queries
- Plan multi-step approaches
- Use available tools (extract_entities, summarize_document, search_documents)
- Chain operations for complex analysis
- Provide comprehensive answers
"""

import json
import re
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import our modules
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from extractor import EntityExtractor
from summarizer import TextSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentTool:
    """Represents a tool available to the agent"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]

@dataclass
class AgentStep:
    """Represents a step in the agent's reasoning process"""
    step_number: int
    action: str
    tool_used: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning: str

class DocumentIntelligenceAgent:
    """Research agent for document analysis and insight generation"""
    
    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []
        self.reasoning_steps = []
        self.context = {}
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor(use_spacy=True)
        self.extractor = EntityExtractor(use_spacy=True, use_transformers=False)
        self.summarizer = TextSummarizer(use_transformers=False)
        
        # Define available tools
        self.tools = {
            'extract_entities': AgentTool(
                name='extract_entities',
                description='Extract entities (people, organizations, dates, etc.) from documents',
                function=self._extract_entities_tool,
                parameters={'document_ids': 'list', 'entity_types': 'list'}
            ),
            'summarize_document': AgentTool(
                name='summarize_document',
                description='Generate summaries of documents using various methods',
                function=self._summarize_document_tool,
                parameters={'document_ids': 'list', 'method': 'str', 'num_sentences': 'int'}
            ),
            'search_documents': AgentTool(
                name='search_documents',
                description='Search for documents matching specific criteria',
                function=self._search_documents_tool,
                parameters={'query': 'str', 'filters': 'dict'}
            ),
            'analyze_trends': AgentTool(
                name='analyze_trends',
                description='Analyze trends and patterns in document entities over time',
                function=self._analyze_trends_tool,
                parameters={'entity_type': 'str', 'time_range': 'str'}
            ),
            'compare_documents': AgentTool(
                name='compare_documents',
                description='Compare multiple documents for similarities and differences',
                function=self._compare_documents_tool,
                parameters={'document_ids': 'list', 'comparison_type': 'str'}
            )
        }
        
        logger.info(f"Agent initialized with {len(self.tools)} available tools")
    
    def load_documents(self, source: str = 'sample', **kwargs):
        """Load documents for analysis"""
        if source == 'reuters':
            self.documents = self.data_loader.load_reuters_corpus(**kwargs)
        elif source == 'custom':
            self.documents = self.data_loader.load_custom_dataset(**kwargs)
        elif source == 'sample':
            self.documents = self.data_loader.create_sample_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown document source: {source}")
        
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def _extract_entities_tool(self, document_ids: List[str] = None, entity_types: List[str] = None) -> Dict:
        """Tool for extracting entities from documents"""
        
        # Select documents
        if document_ids:
            docs_to_process = [doc for doc in self.documents if doc.get('id') in document_ids]
        else:
            docs_to_process = self.documents
        
        if not docs_to_process:
            return {'error': 'No documents found to process'}
        
        # Extract entities
        processed_docs = self.extractor.extract_corpus_entities(docs_to_process)
        
        # Filter by entity types if specified
        results = {'extracted_entities': {}, 'document_count': len(processed_docs)}
        
        for doc in processed_docs:
            doc_id = doc.get('id', 'unknown')
            doc_entities = {}
            
            if 'entities' in doc:
                for method, method_entities in doc['entities'].items():
                    if entity_types:
                        filtered_entities = {k: v for k, v in method_entities.items() if k in entity_types}
                        if filtered_entities:
                            doc_entities[method] = filtered_entities
                    else:
                        doc_entities[method] = method_entities
            
            if doc_entities:
                results['extracted_entities'][doc_id] = doc_entities
        
        return results
    
    def _summarize_document_tool(self, document_ids: List[str] = None, 
                                method: str = 'tfidf', num_sentences: int = 3) -> Dict:
        """Tool for summarizing documents"""
        
        # Select documents
        if document_ids:
            docs_to_process = [doc for doc in self.documents if doc.get('id') in document_ids]
        else:
            docs_to_process = self.documents[:5]  # Limit to first 5 if no IDs specified
        
        if not docs_to_process:
            return {'error': 'No documents found to process'}
        
        # Generate summaries
        summarized_docs = self.summarizer.summarize_corpus(
            docs_to_process, 
            methods=[method], 
            num_sentences=num_sentences
        )
        
        results = {'summaries': {}, 'method': method}
        
        for doc in summarized_docs:
            doc_id = doc.get('id', 'unknown')
            if 'summaries' in doc and method in doc['summaries']:
                summary_data = doc['summaries'][method]
                results['summaries'][doc_id] = {
                    'title': doc.get('title', 'Untitled'),
                    'summary': summary_data.get('summary', ''),
                    'compression_ratio': summary_data.get('compression_ratio', 0.0)
                }
        
        return results
    
    def _search_documents_tool(self, query: str, filters: Dict = None) -> Dict:
        """Tool for searching documents"""
        
        matching_docs = []
        query_lower = query.lower()
        
        for doc in self.documents:
            # Text-based search
            text_content = doc.get('text', '').lower()
            title_content = doc.get('title', '').lower()
            
            if query_lower in text_content or query_lower in title_content:
                match_score = text_content.count(query_lower) + title_content.count(query_lower) * 2
                
                # Apply filters if specified
                if filters:
                    skip_doc = False
                    for filter_key, filter_value in filters.items():
                        if filter_key == 'categories' and 'categories' in doc:
                            if not any(cat in doc['categories'] for cat in filter_value):
                                skip_doc = True
                                break
                        elif filter_key == 'source' and doc.get('source') != filter_value:
                            skip_doc = True
                            break
                        elif filter_key == 'min_length' and len(doc.get('text', '')) < filter_value:
                            skip_doc = True
                            break
                    
                    if skip_doc:
                        continue
                
                matching_docs.append({
                    'id': doc.get('id'),
                    'title': doc.get('title'),
                    'score': match_score,
                    'snippet': self._get_snippet(doc.get('text', ''), query, 100),
                    'length': len(doc.get('text', ''))
                })
        
        # Sort by relevance score
        matching_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'query': query,
            'total_matches': len(matching_docs),
            'documents': matching_docs[:10]  # Return top 10 matches
        }
    
    def _analyze_trends_tool(self, entity_type: str = 'dates', time_range: str = 'all') -> Dict:
        """Tool for analyzing trends in entities over time"""
        
        # First extract entities if not already done
        if not any('entities' in doc for doc in self.documents):
            logger.info("Extracting entities for trend analysis...")
            self.documents = self.extractor.extract_corpus_entities(self.documents)
        
        trends = {'entity_type': entity_type, 'time_range': time_range, 'trends': {}}
        entity_counts = {}
        
        for doc in self.documents:
            if 'entities' not in doc:
                continue
            
            # Look for entities across all methods
            for method, method_entities in doc['entities'].items():
                if entity_type in method_entities:
                    entities = method_entities[entity_type]
                    
                    if isinstance(entities, list):
                        for entity in entities:
                            if isinstance(entity, dict):
                                entity_text = entity.get('text', '')
                            else:
                                entity_text = str(entity)
                            
                            if entity_text:
                                entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
        
        # Sort by frequency
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        
        trends['trends'] = {
            'most_common': sorted_entities[:10],
            'total_unique': len(entity_counts),
            'total_occurrences': sum(entity_counts.values())
        }
        
        return trends
    
    def _compare_documents_tool(self, document_ids: List[str], comparison_type: str = 'entities') -> Dict:
        """Tool for comparing documents"""
        
        docs_to_compare = [doc for doc in self.documents if doc.get('id') in document_ids]
        
        if len(docs_to_compare) < 2:
            return {'error': 'Need at least 2 documents to compare'}
        
        comparison_results = {
            'document_ids': document_ids,
            'comparison_type': comparison_type,
            'results': {}
        }
        
        if comparison_type == 'entities':
            # Extract entities if not already done
            if not any('entities' in doc for doc in docs_to_compare):
                docs_to_compare = self.extractor.extract_corpus_entities(docs_to_compare)
            
            # Compare entities
            all_entities = {}
            for doc in docs_to_compare:
                doc_id = doc.get('id')
                doc_entities = set()
                
                if 'entities' in doc:
                    for method, method_entities in doc['entities'].items():
                        for entity_type, entities in method_entities.items():
                            if isinstance(entities, list):
                                for entity in entities:
                                    if isinstance(entity, dict):
                                        doc_entities.add(entity.get('text', ''))
                                    else:
                                        doc_entities.add(str(entity))
                
                all_entities[doc_id] = doc_entities
            
            # Find common and unique entities
            if len(all_entities) >= 2:
                doc_ids = list(all_entities.keys())
                common_entities = set.intersection(*all_entities.values())
                
                comparison_results['results'] = {
                    'common_entities': list(common_entities),
                    'unique_to_each': {}
                }
                
                for doc_id in doc_ids:
                    others = [all_entities[other_id] for other_id in doc_ids if other_id != doc_id]
                    if others:
                        unique = all_entities[doc_id] - set.union(*others)
                        comparison_results['results']['unique_to_each'][doc_id] = list(unique)
        
        elif comparison_type == 'length':
            # Compare document lengths
            length_comparison = {}
            for doc in docs_to_compare:
                doc_id = doc.get('id')
                length_comparison[doc_id] = {
                    'character_count': len(doc.get('text', '')),
                    'word_count': len(doc.get('text', '').split()),
                    'title': doc.get('title', 'Untitled')
                }
            
            comparison_results['results'] = length_comparison
        
        return comparison_results
    
    def _get_snippet(self, text: str, query: str, max_length: int = 100) -> str:
        """Get a snippet of text around the query term"""
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find the position of the query in the text
        pos = text_lower.find(query_lower)
        if pos == -1:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Calculate snippet boundaries
        start = max(0, pos - max_length // 2)
        end = min(len(text), pos + len(query) + max_length // 2)
        
        snippet = text[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def plan_approach(self, query: str) -> List[Dict]:
        """Plan a multi-step approach to answer the user's query"""
        
        steps = []
        query_lower = query.lower()
        
        # Analyze the query to determine the approach
        if any(keyword in query_lower for keyword in ['summarize', 'summary', 'main points']):
            steps.append({
                'step': 1,
                'action': 'summarize_documents',
                'tool': 'summarize_document',
                'reasoning': 'User is asking for summaries of documents'
            })
        
        if any(keyword in query_lower for keyword in ['entities', 'people', 'organizations', 'companies', 'dates']):
            steps.append({
                'step': len(steps) + 1,
                'action': 'extract_entities',
                'tool': 'extract_entities',
                'reasoning': 'User is asking about entities in the documents'
            })
        
        if any(keyword in query_lower for keyword in ['search', 'find', 'documents about']):
            # Extract search terms
            search_terms = self._extract_search_terms(query)
            steps.append({
                'step': len(steps) + 1,
                'action': 'search_documents',
                'tool': 'search_documents',
                'reasoning': f'User wants to search for documents about: {search_terms}',
                'parameters': {'query': search_terms}
            })
        
        if any(keyword in query_lower for keyword in ['trends', 'patterns', 'over time', 'frequency']):
            steps.append({
                'step': len(steps) + 1,
                'action': 'analyze_trends',
                'tool': 'analyze_trends',
                'reasoning': 'User is asking about trends or patterns'
            })
        
        if any(keyword in query_lower for keyword in ['compare', 'comparison', 'differences', 'similarities']):
            steps.append({
                'step': len(steps) + 1,
                'action': 'compare_documents',
                'tool': 'compare_documents',
                'reasoning': 'User wants to compare documents'
            })
        
        # If no specific steps identified, default to search and summarize
        if not steps:
            search_terms = self._extract_search_terms(query)
            steps = [
                {
                    'step': 1,
                    'action': 'search_documents',
                    'tool': 'search_documents',
                    'reasoning': 'General query - search for relevant documents',
                    'parameters': {'query': search_terms}
                },
                {
                    'step': 2,
                    'action': 'summarize_results',
                    'tool': 'summarize_document',
                    'reasoning': 'Provide summaries of relevant documents'
                }
            ]
        
        return steps
    
    def _extract_search_terms(self, query: str) -> str:
        """Extract key search terms from a query"""
        
        # Simple approach: remove common question words and extract key terms
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'are', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(key_words[:5])  # Return top 5 key words
    
    def execute_plan(self, steps: List[Dict]) -> List[AgentStep]:
        """Execute the planned steps"""
        
        executed_steps = []
        
        for step_info in steps:
            step_number = step_info['step']
            tool_name = step_info['tool']
            reasoning = step_info['reasoning']
            
            logger.info(f"Executing step {step_number}: {reasoning}")
            
            if tool_name not in self.tools:
                logger.error(f"Tool {tool_name} not available")
                continue
            
            try:
                # Prepare inputs
                inputs = step_info.get('parameters', {})
                
                # Execute the tool
                tool = self.tools[tool_name]
                outputs = tool.function(**inputs)
                
                # Create step record
                executed_step = AgentStep(
                    step_number=step_number,
                    action=step_info['action'],
                    tool_used=tool_name,
                    inputs=inputs,
                    outputs=outputs,
                    reasoning=reasoning
                )
                
                executed_steps.append(executed_step)
                
            except Exception as e:
                logger.error(f"Error executing step {step_number}: {e}")
                continue
        
        self.reasoning_steps = executed_steps
        return executed_steps
    
    def synthesize_answer(self, query: str, executed_steps: List[AgentStep]) -> str:
        """Synthesize a comprehensive answer from the executed steps"""
        
        answer_parts = [f"Based on my analysis of the document collection, here's what I found regarding: '{query}'\n"]
        
        for step in executed_steps:
            answer_parts.append(f"\n**{step.action.replace('_', ' ').title()}:**")
            
            if 'summaries' in step.outputs:
                summaries = step.outputs['summaries']
                answer_parts.append(f"Generated summaries for {len(summaries)} documents:")
                for doc_id, summary_data in list(summaries.items())[:3]:  # Show first 3
                    answer_parts.append(f"- {summary_data['title']}: {summary_data['summary'][:200]}...")
            
            elif 'extracted_entities' in step.outputs:
                entities = step.outputs['extracted_entities']
                answer_parts.append(f"Found entities in {len(entities)} documents:")
                # Show sample entities
                for doc_id, doc_entities in list(entities.items())[:2]:
                    for method, method_entities in doc_entities.items():
                        for entity_type, entity_list in list(method_entities.items())[:2]:
                            if entity_list:
                                sample_entities = entity_list[:3] if isinstance(entity_list, list) else [str(entity_list)]
                                answer_parts.append(f"  - {entity_type}: {', '.join(map(str, sample_entities))}")
            
            elif 'documents' in step.outputs:
                docs = step.outputs['documents']
                answer_parts.append(f"Found {step.outputs['total_matches']} matching documents:")
                for doc in docs[:3]:  # Show first 3
                    answer_parts.append(f"- {doc['title']}: {doc['snippet']}")
            
            elif 'trends' in step.outputs:
                trends = step.outputs['trends']
                if 'most_common' in trends:
                    answer_parts.append(f"Top trends in {step.outputs['entity_type']}:")
                    for entity, count in trends['most_common'][:5]:
                        answer_parts.append(f"- {entity}: {count} occurrences")
            
            elif 'results' in step.outputs:
                results = step.outputs['results']
                if 'common_entities' in results:
                    common = results['common_entities']
                    answer_parts.append(f"Found {len(common)} common entities: {', '.join(common[:5])}")
        
        return '\n'.join(answer_parts)
    
    def answer_query(self, query: str) -> str:
        """Main method to answer a user query"""
        
        logger.info(f"Processing query: {query}")
        
        # Plan the approach
        steps = self.plan_approach(query)
        logger.info(f"Planned {len(steps)} steps")
        
        # Execute the plan
        executed_steps = self.execute_plan(steps)
        logger.info(f"Executed {len(executed_steps)} steps")
        
        # Synthesize the answer
        answer = self.synthesize_answer(query, executed_steps)
        
        return answer


def main():
    """Demo function to test the agent"""
    
    # Initialize agent
    agent = DocumentIntelligenceAgent()
    
    # Load sample documents
    agent.load_documents('sample', num_docs=20)
    
    # Example queries
    queries = [
        "What are the main topics in the technology documents?",
        "Find documents about finance and summarize them",
        "What entities appear most frequently across all documents?",
        "Compare the finance and technology documents"
    ]
    
    print("Document Intelligence Agent Demo")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        try:
            answer = agent.answer_query(query)
            print(answer)
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
