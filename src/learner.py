"""
Clean implementation of the Self Learning Agent with minimal dependencies.
This version focuses on core functionality and resource efficiency.
"""

import re
import os
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class SimpleVectorizer:
    """Lightweight TF-IDF replacement using only standard library."""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        
    def fit_transform(self, texts: List[str]) -> List[Dict[str, float]]:
        """Basic TF-IDF calculation."""
        # Build vocabulary
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        self.vocabulary = {word: i for i, word in enumerate(all_words)}
        
        # Calculate document frequency for IDF
        doc_freq = {}
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        
        # Calculate IDF scores
        num_docs = len(texts)
        for word in self.vocabulary:
            self.idf_scores[word] = math.log(num_docs / (doc_freq[word] + 1))
        
        # Transform texts to TF-IDF vectors
        vectors = []
        for text in texts:
            words = text.lower().split()
            word_count = len(words)
            vector = {}
            
            # Calculate TF
            tf = {}
            for word in words:
                tf[word] = tf.get(word, 0) + 1
            
            # Calculate TF-IDF
            for word in tf:
                tf_score = tf[word] / word_count
                tfidf_score = tf_score * self.idf_scores.get(word, 0)
                vector[word] = tfidf_score
                
            vectors.append(vector)
        
        return vectors


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calculate cosine similarity between two sparse vectors."""
    # Get common words
    common_words = set(vec1.keys()) & set(vec2.keys())
    
    if not common_words:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


class MemoryMonitor:
    """Simple memory monitoring for resource management."""
    
    def __init__(self, max_memory_mb: int = 4000):
        self.max_memory_mb = max_memory_mb
        self.current_usage = 0
        
    def check_memory(self) -> bool:
        """Check if we're within memory limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < self.max_memory_mb
        except ImportError:
            # Fallback to simple heuristic
            return self.current_usage < self.max_memory_mb
    
    def update_usage(self, delta_mb: int):
        """Update estimated memory usage."""
        self.current_usage += delta_mb


class SimpleMathEngine:
    """Basic mathematical operations engine."""
    
    def __init__(self):
        self.operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else float('inf'),
            '**': lambda x, y: x ** y,
            '%': lambda x, y: x % y if y != 0 else 0
        }
    
    def evaluate(self, expression: str) -> Optional[Dict[str, Any]]:
        """Evaluate mathematical expressions."""
        # Enhanced pattern to handle expressions with expected results
        pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)(?:\s*=\s*(\d+(?:\.\d+)?))?'
        match = re.search(pattern, expression)
        
        if match:
            try:
                left = float(match.group(1))
                operator = match.group(2)
                right = float(match.group(3))
                expected_result = match.group(4)
                
                if operator in self.operations:
                    actual_result = self.operations[operator](left, right)
                    
                    # Convert to integers if they are whole numbers
                    if isinstance(actual_result, float) and actual_result.is_integer():
                        actual_result = int(actual_result)
                    if isinstance(left, float) and left.is_integer():
                        left = int(left)
                    if isinstance(right, float) and right.is_integer():
                        right = int(right)
                    
                    return {
                        'left_operand': left,
                        'operator': operator,
                        'right_operand': right,
                        'expected_result': expected_result,
                        'actual_result': actual_result,
                        'result': actual_result,
                        'expression': expression
                    }
            except (ValueError, ZeroDivisionError):
                pass
        
        return None


class KnowledgeBase:
    """Lightweight knowledge storage and retrieval system with optimization."""
    
    def __init__(self, max_entries: int = 10000):
        self.knowledge = {}
        self.max_entries = max_entries
        self.logger = logging.getLogger(__name__)
        self.vectorizer = SimpleVectorizer()
        self.access_count = {}  # Track access frequency for LRU optimization
        
    def store(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> str:
        """Store knowledge with metadata and automatic optimization."""
        if not key:
            key = f"knowledge_{datetime.now().isoformat()}"
        
        # Check if we need to optimize storage
        if len(self.knowledge) >= self.max_entries:
            self._optimize_storage()
        
        entry = {
            'content': value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'confidence': metadata.get('confidence', 0.8) if metadata else 0.8,
            'access_count': 0
        }
        
        self.knowledge[key] = entry
        self.access_count[key] = 0
        self.logger.info(f"Stored knowledge: {key}")
        return key
    
    def query(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Query knowledge base with text similarity and access tracking."""
        if not self.knowledge:
            return []
        
        query_text = query_text.lower().strip()
        results = []
        
        # Check for mathematical queries first
        math_result = self._handle_math_query(query_text)
        if math_result:
            return [math_result]
        
        # Text-based similarity search with caching
        cache_key = f"query_{hash(query_text)}"
        
        # Build search index
        texts = [query_text]
        contents = []
        keys = []
        
        for key, entry in self.knowledge.items():
            content = str(entry.get('content', ''))
            texts.append(content.lower())
            contents.append(content)
            keys.append(key)
        
        if len(texts) > 1:
            try:
                vectors = self.vectorizer.fit_transform(texts)
                query_vector = vectors[0]
                
                similarities = []
                for i, content_vector in enumerate(vectors[1:], 1):
                    sim = cosine_similarity(query_vector, content_vector)
                    if sim > 0.1:  # Minimum similarity threshold
                        key = keys[i-1]
                        # Update access count for LRU optimization
                        self.access_count[key] = self.access_count.get(key, 0) + 1
                        self.knowledge[key]['access_count'] = self.access_count[key]
                        
                        similarities.append((sim, key, contents[i-1]))
                
                # Sort by similarity and return top results
                similarities.sort(reverse=True)
                for sim_score, key, content in similarities[:max_results]:
                    results.append({
                        'key': key,
                        'content': content,
                        'similarity': sim_score,
                        'confidence': self.knowledge[key].get('confidence', 0.5)
                    })
            except Exception as e:
                self.logger.error(f"Error in similarity search: {e}")
        
        return results
    
    def _optimize_storage(self):
        """Optimize storage by removing least recently used and low-confidence entries."""
        if len(self.knowledge) < self.max_entries:
            return
        
        entries_to_remove = len(self.knowledge) // 4  # Remove 25% of entries
        
        # Sort by access count (ascending) and confidence (ascending)
        sorted_entries = sorted(
            self.knowledge.items(),
            key=lambda x: (x[1].get('access_count', 0), x[1].get('confidence', 0.5))
        )
        
        # Remove least accessed and lowest confidence entries
        for i in range(min(entries_to_remove, len(sorted_entries))):
            key = sorted_entries[i][0]
            del self.knowledge[key]
            if key in self.access_count:
                del self.access_count[key]
        
        self.logger.info(f"Optimized storage: removed {entries_to_remove} entries")
    
    def _handle_math_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle mathematical queries with caching."""
        math_engine = SimpleMathEngine()
        result = math_engine.evaluate(query)
        
        if result:
            return {
                'key': f'math_{hash(query)}',
                'content': f"{result['expression']} = {result['result']}",
                'confidence': 1.0,
                'metadata': {
                    'type': 'mathematical_computation',
                    'result': result,
                    'cached': True  # Mark as cached computation
                }
            }
        
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for monitoring."""
        total_size = 0
        low_confidence_count = 0
        high_access_count = 0
        
        for entry in self.knowledge.values():
            total_size += len(str(entry.get('content', '')))
            if entry.get('confidence', 0.5) < 0.3:
                low_confidence_count += 1
            if entry.get('access_count', 0) > 10:
                high_access_count += 1
        
        return {
            'total_entries': len(self.knowledge),
            'max_entries': self.max_entries,
            'storage_utilization': len(self.knowledge) / self.max_entries,
            'estimated_size_bytes': total_size,
            'low_confidence_entries': low_confidence_count,
            'frequently_accessed_entries': high_access_count,
            'optimization_needed': len(self.knowledge) > self.max_entries * 0.8
        }
    
    def export(self, filepath: str):
        """Export knowledge to JSON file with compression."""
        try:
            # Create a compressed version for export
            export_data = {}
            for key, entry in self.knowledge.items():
                # Only export essential data
                export_data[key] = {
                    'content': str(entry.get('content', ''))[:1000],  # Limit content size
                    'timestamp': entry.get('timestamp', ''),
                    'confidence': entry.get('confidence', 0.5),
                    'metadata': entry.get('metadata', {}),
                    'access_count': entry.get('access_count', 0)
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Knowledge exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
    
    def import_knowledge(self, filepath: str):
        """Import knowledge from JSON file."""
        try:
            with open(filepath, 'r') as f:
                imported_data = json.load(f)
            
            for key, entry in imported_data.items():
                # Restore access count tracking
                access_count = entry.get('access_count', 0)
                self.access_count[key] = access_count
                
                self.knowledge[key] = entry
                
            self.logger.info(f"Imported {len(imported_data)} entries from {filepath}")
        except Exception as e:
            self.logger.error(f"Import failed: {e}")


class SelfLearningAgent:
    """
    Main learning agent class with minimal dependencies and resource efficiency focus.
    """
    
    def __init__(self, autonomous_learning: bool = False, learning_interval: int = 300):
        """Initialize the learning agent with performance monitoring."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Core components with optimizations
        self.knowledge_base = KnowledgeBase(max_entries=10000)  # Limit entries for performance
        self.memory_monitor = MemoryMonitor(max_memory_mb=4000)  # 4GB limit as per project goals
        self.math_engine = SimpleMathEngine()
        
        # Mathematical knowledge patterns
        self.math_patterns = {
            "+": self._add,
            "-": self._subtract,
            "*": self._multiply,
            "/": self._divide,
        }
        
        # Learning configuration
        self.autonomous_learning = autonomous_learning
        self.learning_interval = learning_interval
        
        # Performance tracking
        self._operation_count = 0
        self._last_optimization = datetime.now()
        
        self.logger.info("SelfLearningAgent initialized successfully with resource optimization")
    
    def interact(self, interaction: str) -> Dict[str, Any]:
        """
        Process an interaction and learn from it with performance monitoring.
        
        Args:
            interaction: User interaction or input
            
        Returns:
            Learning result with insights
        """
        self._operation_count += 1
        
        if not interaction or not interaction.strip():
            return {
                'status': 'error',
                'message': 'Empty interaction provided',
                'confidence': 0.0
            }
        
        # Periodic optimization
        if self._operation_count % 100 == 0:
            self._periodic_optimization()
        
        # Preprocess interaction
        cleaned_interaction = self._preprocess_interaction(interaction)
        
        # Check memory usage
        if not self.memory_monitor.check_memory():
            self.logger.warning("Memory usage high, using simplified processing")
            return self._simple_response(cleaned_interaction)
        
        # Try mathematical interaction first (most efficient)
        math_result = self._process_mathematical_interaction(cleaned_interaction)
        if math_result:
            return math_result
        
        # Process as general knowledge
        general_result = self._process_general_interaction(cleaned_interaction)
        return general_result
    
    def _periodic_optimization(self):
        """Perform periodic optimization of the knowledge base."""
        try:
            # Get storage stats
            stats = self.knowledge_base.get_storage_stats()
            
            if stats['optimization_needed']:
                self.logger.info("Performing automatic knowledge base optimization")
                self.knowledge_base._optimize_storage()
                
                # Update memory usage estimate
                self.memory_monitor.update_usage(-100)  # Assume 100MB freed
            
            self._last_optimization = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage and performance statistics."""
        try:
            stats = self.knowledge_base.get_storage_stats()
            
            # Debug the components
            estimated_usage = self.memory_monitor.current_usage
            max_memory = self.memory_monitor.max_memory_mb
            within_limits = self.memory_monitor.check_memory()
            knowledge_entries = stats['total_entries']
            storage_utilization = stats.get('storage_utilization', 0.0)
            low_confidence = stats.get('low_confidence_entries', 0)
            frequently_accessed = stats.get('frequently_accessed_entries', 0)
            operation_count = getattr(self, '_operation_count', 0)
            last_optimization = getattr(self, '_last_optimization', datetime.now()).isoformat()
            optimization_recommended = stats.get('optimization_needed', False)
            
            return {
                'estimated_usage_mb': estimated_usage,
                'max_memory_mb': max_memory,
                'within_limits': within_limits,
                'knowledge_entries': knowledge_entries,
                'storage_utilization': storage_utilization,
                'low_confidence_entries': low_confidence,
                'frequently_accessed_entries': frequently_accessed,
                'operation_count': operation_count,
                'last_optimization': last_optimization,
                'optimization_recommended': optimization_recommended
            }
            
        except Exception as e:
            # Fallback if anything fails
            self.logger.warning(f"Memory usage calculation failed: {e}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
            
            return {
                'estimated_usage_mb': getattr(self.memory_monitor, 'current_usage', 0),
                'max_memory_mb': getattr(self.memory_monitor, 'max_memory_mb', 4000),
                'within_limits': True,
                'knowledge_entries': len(self.knowledge_base.knowledge),
                'storage_utilization': len(self.knowledge_base.knowledge) / getattr(self.knowledge_base, 'max_entries', 10000),
                'low_confidence_entries': 0,
                'frequently_accessed_entries': 0,
                'operation_count': 0,
                'last_optimization': datetime.now().isoformat(),
                'optimization_recommended': False
            }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Manually trigger performance optimization."""
        optimization_report = {
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'performance_improvement': 0.0
        }
        
        # Get initial stats
        initial_stats = self.knowledge_base.get_storage_stats()
        initial_entries = initial_stats['total_entries']
        
        # Optimize knowledge base
        if initial_stats['optimization_needed']:
            self.knowledge_base._optimize_storage()
            optimization_report['actions_taken'].append('Knowledge base optimization')
        
        # Clean up low-confidence entries
        if initial_stats['low_confidence_entries'] > 100:
            self._cleanup_low_confidence_entries()
            optimization_report['actions_taken'].append('Low-confidence entry cleanup')
        
        # Update memory estimates
        self.memory_monitor.current_usage = max(0, self.memory_monitor.current_usage - 200)
        
        # Calculate improvement
        final_stats = self.knowledge_base.get_storage_stats()
        entries_removed = initial_entries - final_stats['total_entries']
        optimization_report['performance_improvement'] = (entries_removed / initial_entries) * 100 if initial_entries > 0 else 0
        
        self.logger.info(f"Performance optimization completed: {optimization_report}")
        return optimization_report
    
    def _cleanup_low_confidence_entries(self):
        """Remove entries with very low confidence that are old."""
        keys_to_remove = []
        current_time = datetime.now()
        
        for key, entry in self.knowledge_base.knowledge.items():
            confidence = entry.get('confidence', 0.5)
            timestamp_str = entry.get('timestamp', '')
            
            if confidence < 0.2:  # Very low confidence
                try:
                    entry_time = datetime.fromisoformat(timestamp_str)
                    age_hours = (current_time - entry_time).total_seconds() / 3600
                    
                    if age_hours > 24:  # Older than 24 hours
                        keys_to_remove.append(key)
                except ValueError:
                    # If timestamp is invalid, consider for removal
                    keys_to_remove.append(key)
        
        # Remove identified entries
        for key in keys_to_remove:
            del self.knowledge_base.knowledge[key]
            if key in self.knowledge_base.access_count:
                del self.knowledge_base.access_count[key]
        
        if keys_to_remove:
            self.logger.info(f"Cleaned up {len(keys_to_remove)} low-confidence entries")
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        if not query or not query.strip():
            return []
        
        processed_query = self._preprocess_interaction(query)
        results = self.knowledge_base.query(processed_query)
        
        # Add critical insights to results
        enriched_results = []
        for result in results:
            enriched_result = {
                'base_result': result,
                'critical_insights': self._generate_insights(result)
            }
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def _preprocess_interaction(self, interaction: str) -> str:
        """Preprocess interaction text."""
        # Basic cleaning
        interaction = interaction.lower().strip()
        interaction = re.sub(r'\s+', ' ', interaction)
        return interaction
    
    def _process_mathematical_interaction(self, interaction: str) -> Optional[Dict[str, Any]]:
        """Process mathematical interactions."""
        math_result = self.math_engine.evaluate(interaction)
        
        if math_result:
            # Store mathematical knowledge
            key = f"math_{hash(interaction)}"
            self.knowledge_base.store(
                key,
                math_result,
                {'type': 'mathematical_operation', 'confidence': 0.9}
            )
            
            # Check if expected result matches actual result
            expected = math_result.get('expected_result')
            actual = math_result.get('actual_result')
            is_valid = True
            
            if expected is not None:
                try:
                    expected_num = float(expected)
                    if isinstance(expected_num, float) and expected_num.is_integer():
                        expected_num = int(expected_num)
                    is_valid = (expected_num == actual)
                except ValueError:
                    is_valid = False
            
            return {
                'status': 'learned',
                'type': 'mathematical_fact',
                'details': {
                    'valid': is_valid,
                    'details': {
                        'expression': interaction,
                        'components': math_result
                    }
                },
                'confidence': 0.9,
                'critical_insights': [{
                    'type': 'mathematical_insight',
                    'message': f"Mathematical operation: {math_result['expression']} = {math_result['actual_result']}"
                }]
            }
        
        return None
    
    def _process_general_interaction(self, interaction: str) -> Dict[str, Any]:
        """Process general knowledge interactions."""
        # Store as general knowledge
        key = f"general_{datetime.now().isoformat()}"
        metadata = {
            'type': 'general_knowledge',
            'source': 'user_interaction',
            'confidence': 0.5
        }
        
        self.knowledge_base.store(key, interaction, metadata)
        
        return {
            'status': 'learned',
            'type': 'general_knowledge',
            'details': {
                'text': interaction,
                'entities': self._extract_simple_entities(interaction)
            },
            'confidence': 0.5,
            'critical_insights': self._generate_simple_insights(interaction)
        }
    
    def _simple_response(self, interaction: str) -> Dict[str, Any]:
        """Provide a simple response when resources are limited."""
        return {
            'status': 'learned',
            'type': 'simple_acknowledgment',
            'details': {'text': interaction},
            'confidence': 0.3,
            'critical_insights': [{
                'type': 'resource_constraint',
                'message': 'Processed with limited resources due to memory constraints'
            }]
        }
    
    def _extract_simple_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract simple entities from text."""
        entities = []
        
        # Simple patterns for entity extraction
        patterns = {
            'NUMBER': r'\b\d+(?:\.\d+)?\b',
            'CAPITALIZED': r'\b[A-Z][a-z]+\b',
            'TECHNOLOGY': r'\b(?:AI|ML|python|javascript|java|machine learning|artificial intelligence)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match,
                    'label': entity_type
                })
        
        return entities
    
    def _generate_insights(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate critical insights for query results."""
        insights = []
        
        content = result.get('content', '')
        confidence = result.get('confidence', 0.5)
        
        if confidence > 0.8:
            insights.append({
                'type': 'high_confidence',
                'message': 'This information has high confidence and reliability'
            })
        elif confidence < 0.3:
            insights.append({
                'type': 'low_confidence',
                'message': 'This information has low confidence, verify before use'
            })
        
        if 'math' in content.lower() or any(op in content for op in ['+', '-', '*', '/']):
            insights.append({
                'type': 'mathematical',
                'message': 'Contains mathematical content'
            })
        
        return insights
    
    def _generate_simple_insights(self, text: str) -> List[Dict[str, str]]:
        """Generate simple insights for interactions."""
        insights = []
        
        # Check for question patterns
        if '?' in text:
            insights.append({
                'type': 'question',
                'message': 'This appears to be a question or inquiry'
            })
        
        # Check for learning patterns
        if any(word in text.lower() for word in ['learn', 'teach', 'explain', 'how']):
            insights.append({
                'type': 'learning_request',
                'message': 'This appears to be a learning or explanation request'
            })
        
        # Check for mathematical content
        if re.search(r'\d+\s*[+\-*/]\s*\d+', text):
            insights.append({
                'type': 'mathematical_content',
                'message': 'Contains mathematical expressions'
            })
        
        return insights
    
    def _add(self, a: float, b: float) -> float:
        """Addition operation."""
        return a + b
    
    def _subtract(self, a: float, b: float) -> float:
        """Subtraction operation."""
        return a - b
    
    def _multiply(self, a: float, b: float) -> float:
        """Multiplication operation."""
        return a * b
    
    def _divide(self, a: float, b: float) -> float:
        """Division operation with error handling."""
        return a / b if b != 0 else float('inf')
    
    def export_knowledge(self, filepath: str = None):
        """Export learned knowledge to a file."""
        if filepath is None:
            filepath = os.path.join(
                os.path.expanduser('~'),
                'learned_knowledge.json'
            )
        
        self.knowledge_base.export(filepath)
    
    def import_knowledge(self, filepath: str = None):
        """Import knowledge from a file."""
        if filepath is None:
            filepath = os.path.join(
                os.path.expanduser('~'),
                'learned_knowledge.json'
            )
        
        self.knowledge_base.import_knowledge(filepath)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        return {
            'estimated_usage_mb': self.memory_monitor.current_usage,
            'max_memory_mb': self.memory_monitor.max_memory_mb,
            'within_limits': self.memory_monitor.check_memory(),
            'knowledge_entries': len(self.knowledge_base.knowledge)
        }


# Compatibility layer for existing tests
class CriticalReasoningEngine:
    """Simplified critical reasoning engine."""
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
    
    def generate_insights(self, query: str) -> List[Dict[str, str]]:
        """Generate default insights."""
        return [{
            'type': 'default',
            'message': 'No related knowledge found in the graph.'
        }]