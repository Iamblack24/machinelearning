import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import logging
from .knowledge_graph import KnowledgeGraph

class KnowledgeBase:
    def __init__(self, initial_knowledge: Dict[str, Any] = None):
        """
        Initialize the knowledge base with optional initial knowledge.
        
        :param initial_knowledge: Optional dictionary of initial knowledge
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Add a handler if no handler exists
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize knowledge storage
        self.knowledge = initial_knowledge or {}
        
        # Initialize learning history
        self.learning_history: List[Dict[str, Any]] = []
        
        # Confidence tracking
        self.confidence_decay_rate = 0.01   

        # New attribute for persistent knowledge
        self.persistent_knowledge = {}
        
        # Load persistent knowledge on initialization
        self._load_persistent_knowledge()
        
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()

        # Ephemeral file in project root to store in-memory knowledge;
        # this file is separate from persistent_knowledge.json.
        self.ephemeral_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ephemeral_knowledge.json")
        self._load_ephemeral_knowledge()

    def store(self, key: str, value: Any, context: Any = None):
        """Store knowledge with standardized format"""
        if not key:
            key = f"knowledge_{datetime.now().isoformat()}"

        knowledge_entry = {
            'content': str(value) if isinstance(value, (str, int, float)) else value.get('content', ''),
            'timestamp': datetime.now().isoformat(),
            'confidence': context.get('confidence', 0.8) if context else 0.8,
            'metadata': {
                'source': context.get('source', 'user_input') if context else 'user_input',
                'type': context.get('type', 'general') if context else 'general',
                'entities': context.get('entities', []) if context else []
            }
        }
        
        self.knowledge[key] = knowledge_entry
        self.save_ephemeral()  # Save to ephemeral file
        self.save_persistent()  # Save to persistent file
        self.logger.info(f'Storing knowledge: {key}')
        return key

    def retrieve(self, key: str) -> Dict[str, Any]:
        """Retrieve knowledge with its metadata."""
        return self.knowledge.get(key, None)
    

    def update_confidence(self, key: str, confidence_delta: float):
        """
        Enhanced confidence update with decay and validation mechanisms
        
        :param key: Knowledge key
        :param confidence_delta: Change in confidence
        """
        if key not in self.knowledge:
            return

        current_confidence = self.knowledge[key].get('confidence', 0.5)
        
        # Apply confidence update with bounds
        new_confidence = max(0, min(1, current_confidence + confidence_delta))
        
        # Apply confidence decay over time
        time_since_learn = datetime.now() - datetime.fromisoformat(
            self.knowledge[key].get('timestamp', datetime.now().isoformat())
        )
        
        # Decay confidence based on time (more decay for older knowledge)
        decay_factor = max(0.1, 1 - (time_since_learn.days * 0.01))
        new_confidence *= decay_factor
        
        self.knowledge[key]['confidence'] = new_confidence

    def export(self, filepath: str):
        """
        Export knowledge to a JSON file.
        
        :param filepath: Path to export knowledge
        """
        # Create a flattened, serializable version of knowledge
        exportable_knowledge = {}
        for key, entry in self.knowledge.items():
            exportable_knowledge[key] = {
                'value': str(entry.get('value', '')),
                'context': str(entry.get('context', {})),
                'confidence': float(entry.get('confidence', 0.5))
            }
        
        with open(filepath, 'w') as f:
            json.dump(exportable_knowledge, f, indent=2)

    def import_knowledge(self, filepath: str):
        """
        Import knowledge from a JSON file with robust parsing.
        
        :param filepath: Path to the JSON file
        """
        try:
            with open(filepath, 'r') as f:
                imported_data = json.load(f)
            
            # Handle different import formats
            for key, entry in imported_data.items():
                # Normalize context
                context = {}
                if isinstance(entry, dict):
                    # If entry is a dictionary, try to extract context
                    context = entry.get('context', {})
                    value = entry.get('value', entry)
                else:
                    # Simple value
                    value = entry
                
                # Normalize context to dictionary
                if not isinstance(context, dict):
                    context = {'original_type': type(context).__name__}
                
                # Store with normalized structure
                self.store(
                    key, 
                    value, 
                    context
                )
        
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error importing knowledge: {e}")
            raise ValueError(f"Could not import knowledge from {filepath}: {e}")

    def query(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Enhanced query with math expression handling and better text matching."""
        query_text = query_text.lower().strip()
        
        # Check for mathematical expression
        math_result = self._process_math_query(query_text)
        if math_result:
            return [math_result]

        # Continue with normal text-based query...
        results = []

        # Refresh persistent knowledge from file to capture updates
        self._load_persistent_knowledge()

        # Combine in-memory and persistent knowledge
        combined_knowledge = {**self.knowledge, **self.persistent_knowledge}
        
        for key, entry in combined_knowledge.items():
            # Extract content: if dictionary, try common keys
            content_raw = entry.get('content', '')
            if isinstance(content_raw, dict):
                content = (content_raw.get('text') or 
                           content_raw.get('summary') or 
                           content_raw.get('topic') or 
                           str(content_raw))
            else:
                content = str(content_raw)
            content_lower = content.lower()

            # Calculate similarity using TF-IDF-based cosine similarity
            similarity = self._calculate_similarity(query_text, content_lower)

            # Fallback: if the query substring exists, consider it a match
            if similarity > 0.3 or query_text in content_lower:
                results.append({
                    'key': key,
                    'content': content,
                    'confidence': entry.get('confidence', 0.8),
                    'metadata': entry.get('metadata', {})
                })
        return sorted(results, key=lambda x: x['confidence'], reverse=True)[:max_results]

    def _process_math_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Process mathematical queries and return computed results."""
        import re
        
        # Extract numbers and operator from queries like "what is 2+5" or "2 + 5"
        math_pattern = r'(?:what\s+is\s+)?(\d+)\s*([\+\-\*\/])\s*(\d+)'
        match = re.search(math_pattern, query)
        
        if match:
            num1 = int(match.group(1))
            operator = match.group(2)
            num2 = int(match.group(3))
            
            result = None
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/' and num2 != 0:
                result = num1 / num2
            
            if result is not None:
                return {
                    'key': f'math_{hash(query)}',
                    'content': f"{num1} {operator} {num2} = {result}",
                    'confidence': 1.0,
                    'metadata': {
                        'type': 'mathematical_computation',
                        'operation': operator,
                        'operands': [num1, num2],
                        'result': result
                    }
                }
        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score using TF-IDF and cosine similarity."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
            
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0

    def _load_persistent_knowledge(self):
        """
        Load persistent knowledge from the project root's persistent_knowledge.json.
        """
        try:
            persistent_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'persistent_knowledge.json')
            if os.path.exists(persistent_path):
                with open(persistent_path, 'r') as f:
                    loaded_knowledge = json.load(f)
                # Use the loaded structure as persistent knowledge
                self.persistent_knowledge = loaded_knowledge
                self.logger.info(f"Loaded {len(loaded_knowledge)} persistent knowledge entries")
        except Exception as e:
            self.logger.error(f"Failed to load persistent knowledge: {e}")

    def save_ephemeral(self):
        """Save only ephemeral/in-memory knowledge"""
        try:
            with open(self.ephemeral_file, "w") as f:
                json.dump(self.knowledge, f, indent=2)
            self.logger.info(f"Ephemeral knowledge saved to {self.ephemeral_file}")
        except Exception as e:
            self.logger.error(f"Error saving ephemeral knowledge: {e}")

    async def async_save_ephemeral(self):
        """
        Asynchronously save only ephemeral (in-memory) knowledge.
        """
        try:
            await asyncio.to_thread(self._sync_save_ephemeral)
            self.logger.info(f"Ephemeral knowledge saved to {self.ephemeral_file}")
        except Exception as e:
            self.logger.error(f"Error saving ephemeral knowledge: {e}")

    def _sync_save_ephemeral(self):
        with open(self.ephemeral_file, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    async def async_save_persistent(self, filepath: Optional[str] = None):
        """
        Asynchronously save knowledge to persistent storage.
        """
        try:
            await asyncio.to_thread(self._sync_save_persistent, filepath)
            self.logger.info(f"Knowledge saved to {filepath if filepath else 'persistent_knowledge.json'}")
        except Exception as e:
            self.logger.error(f"Failed to save persistent knowledge: {e}")

    def _sync_save_persistent(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'persistent_knowledge.json')
        existing = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing = json.load(f)
        combined = {**existing, **self.knowledge}
        with open(filepath, 'w') as f:
            json.dump(combined, f, indent=2)

    def save_persistent(self, filepath: Optional[str] = None):
        """Save to persistent storage in project root"""
        try:
            if filepath is None:
                filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'persistent_knowledge.json')
            
            # Load existing persistent knowledge first
            existing = {}
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            
            # Merge with new knowledge
            combined = {**existing, **self.knowledge}
            
            with open(filepath, 'w') as f:
                json.dump(combined, f, indent=2)
            self.logger.info(f"Knowledge saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save persistent knowledge: {e}")

    def values(self):
        """
        Return all knowledge values.
        
        :return: List of all knowledge values
        """
        # Combine knowledge and persistent knowledge
        combined_knowledge = {**self.knowledge, **self.persistent_knowledge}
        return list(combined_knowledge.values())

    def _load_ephemeral_knowledge(self):
        try:
            if os.path.exists(self.ephemeral_file):
                with open(self.ephemeral_file, "r") as f:
                    self.knowledge = json.load(f)
                    self.logger.info(f"Loaded ephemeral knowledge from {self.ephemeral_file}")
        except Exception as e:
            self.logger.error(f"Error loading ephemeral knowledge: {e}")