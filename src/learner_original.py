import re
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import KnowledgeBase
from .autonomous_learner import AutonomousLearningAgent
from .critical_reasoning import CriticalReasoningEngine
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class FoundationalTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear_out(output)
        return output

class SelfLearningAgent:
    def __init__(self, vocab_size=50000, d_model=512):
        # Initialize our custom foundational model
        self.model = FoundationalTransformer(
            vocab_size=vocab_size,
            d_model=d_model
        )
        
        # Initialize tokenizer and vocabulary
        self.tokenizer = CustomTokenizer(vocab_size)
        
        # Knowledge management
        self.knowledge_base = KnowledgeBase()
        self.critical_reasoning = CriticalReasoningEngine(self.knowledge_base)
        
        # Training configuration
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
        # Replace spaCy with distilled model
        self.nlp = pipeline(
            "text-generation",
            model="microsoft/phi-2",
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # Add memory-efficient knowledge graph
        self.knowledge_graph = KnowledgeGraph(
            embedding_size=128,  # Reduced from 512
            sparse_embeddings=True
        )
        
        # Add model quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            load_in_4bit=True,  # For 4-bit quantization
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        
        # Initialize core components
        self.knowledge_base = KnowledgeBase()
        self.autonomous_learner = AutonomousLearningAgent(learning_interval)

        # Load the NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer()
        self.knowledge_graph = self.knowledge_base.knowledge_graph

        # Mathematical knowledge representation
        self.math_patterns = {
            "+": self._add,
            "-": self._subtract,
            "*": self._multiply,
            "/": self._divide,
        }

        # Critical Reasoning Engine
        self.critical_reasoning = CriticalReasoningEngine(self.knowledge_base)

        # Autonomous learning setup
        self.autonomous_agent = None
        if autonomous_learning:
            self.autonomous_agent = AutonomousLearningAgent(learning_interval)
            self.autonomous_agent.start_learning()
        
        # access knowledgeGraph
        self.knowledge_graph = self.knowledge_base.knowledge_graph

        self.math_operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else float('inf')
        }

    def interact(self, interaction: str) -> Dict[str, Any]:
        """
        Process an interaction and learn from it.

        :param interaction: User interaction or input
        :return: Learning result with critical insights
        """
        if self.autonomous_agent:
            self.autonomous_agent.learn(interaction)
        
        # Preprocess interaction
        cleaned_interaction = self._preprocess_interaction(interaction)
        
        # Store the cleaned interaction in the knowledge base
        self.knowledge_base.store(key='interaction', value=cleaned_interaction)

        # Try mathematical interaction first
        math_result = self._process_mathematical_interaction(cleaned_interaction)
        if math_result:
            # Generate insights for math
            critical_insights = self.critical_reasoning.generate_insights(cleaned_interaction)
            math_result['critical_insights'] = critical_insights
            
            # Ensure confidence is part of the result
            math_result['base_result'] = {
                **math_result.get('details', {}),
                'confidence': math_result.get('confidence', 0.8)
            }
            
            return math_result
        
        # Try general interaction
        general_result = self._process_general_interaction(cleaned_interaction)
        if general_result:
            # Generate insights for general interaction
            critical_insights = self.critical_reasoning.generate_insights(cleaned_interaction)
            general_result['critical_insights'] = critical_insights
            
            # Ensure confidence is part of the result
            general_result['base_result'] = {
                **general_result.get('details', {}),
                'confidence': 0.5  # Default confidence
            }
            
            return general_result
        
        # Fallback: store as generic knowledge
        generic_key = f"generic_{datetime.now().isoformat()}"
        self.knowledge_base.store(
            generic_key, 
            cleaned_interaction, 
            {'type': 'unclassified_knowledge'}
        )

        # Generate insights for generic knowledge
        critical_insights = self.critical_reasoning.generate_insights(cleaned_interaction)
        
        return {
            'status': 'learned',
            'type': 'generic_knowledge',
            'details': {
                'text': cleaned_interaction
            },
            'base_result': {
                'text': cleaned_interaction,
                'confidence': 0.3  # Low confidence for generic knowledge
            },
            'critical_insights': critical_insights
        }

    def query(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced query method with mathematical expression support."""
        processed_query = self._preprocess_interaction(query)
        
        # Get results from knowledge base (which now handles math expressions)
        base_results = self.knowledge_base.query(processed_query)
        
        if not base_results:
            return []

        # Generate critical insights for non-mathematical queries
        if not any(r.get('metadata', {}).get('type') == 'mathematical_computation' 
                  for r in base_results):
            critical_insights = self.critical_reasoning.generate_insights(processed_query)
        else:
            critical_insights = [{
                'type': 'mathematical_insight',
                'message': 'Direct mathematical computation performed.'
            }]

        # Enrich results
        return [{
            'base_result': result,
            'critical_insights': critical_insights
        } for result in base_results]

    def _preprocess_interaction(self, interaction: str) -> str:
        """
        Preprocess interaction text.

        :param interaction: Input interaction
        :return: Preprocessed interaction
        """
        # Convert to lowercase
        interaction = interaction.lower().strip()
        
        # Remove extra whitespaces
        interaction = re.sub(r'\s+', ' ', interaction)
        
        return interaction

    def _process_mathematical_interaction(self, interaction: str):
        """
        Process mathematical interactions with advanced learning.

        :param interaction: Interaction text
        :return: Mathematical operation result or None
        """
        # Enhanced regex for mathematical operations
        pattern = r'(\d+)\s*([+\-*/])\s*(\d+)\s*=?\s*(\d+)?'
        match = re.match(pattern, interaction)
        
        if match:
            left_operand = int(match.group(1))
            operator = match.group(2)
            right_operand = int(match.group(3))
            
            # Compute result using predefined math patterns
            result_func = self.math_patterns.get(operator)
            if result_func:
                result = result_func(left_operand, right_operand)
                
                # Validate result if expected result is provided
                expected_result = match.group(4)
                is_valid = (
                    expected_result is None or 
                    int(expected_result) == result
                )
                
                # Create mathematical fact with generic pattern learning
                math_fact = {
                    'status': 'learned',
                    'type': 'mathematical_fact',
                    'details': {
                        'valid': is_valid,
                        'details': {
                            'expression': interaction,
                            'components': {
                                'left_operand': left_operand,
                                'operator': operator,
                                'right_operand': right_operand,
                                'expected_result': expected_result or str(result),
                                'actual_result': result
                            }
                        }
                    },
                    'confidence': 0.8  # High confidence for mathematical facts
                }
                
                # Store generic mathematical pattern
                generic_pattern_key = f"math_pattern_{operator}"
                generic_pattern = {
                    'operator': operator,
                    'description': f"Mathematical operation using {operator}",
                    'example': interaction
                }
                
                # Store both specific instance and generic pattern
                self.knowledge_base.store(
                    f"math_{left_operand}_{operator}_{right_operand}", 
                    math_fact,
                    {'type': 'mathematical_operation'}
                )
                self.knowledge_base.store(
                    generic_pattern_key, 
                    generic_pattern,
                    {'type': 'mathematical_pattern'}
                )
                
                return math_fact
        
        return None

    def _process_general_interaction(self, interaction: str) -> Dict[str, Any]:
        """
        Process general knowledge interactions.

        :param interaction: User interaction text
        :return: Processed interaction result
        """
        # Identify entities and linguistic properties
        linguistic_result = self._analyze_linguistic_properties(interaction)
        
        # Store general knowledge
        key = f"interaction_{int(datetime.now().timestamp())}"
        general_knowledge = {
            'content': {
                'input': interaction,
                'result': linguistic_result
            },
            'timestamp': datetime.now().timestamp(),
            'source': 'user_interaction'
        }
        
        # Store in knowledge base
        self.knowledge_base.store(
            key, 
            general_knowledge, 
            {'type': 'general_knowledge', 'source': 'user_interaction'}
        )
        
        # Return processed result with confidence
        return {
            'status': 'learned',
            'type': 'general_knowledge',
            'details': {
                'text': interaction,
                'entities': linguistic_result.get('entities', [])
            },
            'base_result': {
                'text': interaction,
                'entities': linguistic_result.get('entities', []),
                'confidence': 0.5  # Default confidence for general knowledge
            }
        }

    def _analyze_linguistic_properties(self, interaction: str) -> Dict[str, Any]:
        """
        Analyze linguistic properties of an interaction using SpaCy.

        :param interaction: Input text to analyze
        :return: Linguistic analysis result
        """
        # Use SpaCy for entity recognition
        doc = self.nlp(interaction)
        
        # Extract entities
        entities = [
            {
                'text': ent.text, 
                'label': ent.label_
            } for ent in doc.ents
        ]
        
        # Determine linguistic type
        linguistic_type = 'linguistic_input'
        
        return {
            'status': 'processed',
            'type': linguistic_type,
            'entities': entities
        }

    def _is_related(self, result: Dict, insight: Dict) -> bool:
        """
        Determine if an insight is related to a result.

        :param result: Base result
        :param insight: Critical insight
        :return: Boolean indicating relatedness
        """
        # Simple relatedness check
        result_text = str(result).lower()
        insight_text = str(insight).lower()
        
        return any(
            term in result_text 
            for term in insight_text.split()
        )

    def _add(self, a: int, b: int) -> int:
        """Addition operation."""
        return a + b

    def _subtract(self, a: int, b: int) -> int:
        """Subtraction operation."""
        return a - b

    def _multiply(self, a: int, b: int) -> int:
        """Multiplication operation."""
        return a * b

    def _divide(self, a: int, b: int) -> float:
        """Division operation with error handling."""
        try:
            return a / b
        except ZeroDivisionError:
            return float('inf')

    def export_knowledge(self, filepath: str = None):
        """
        Export learned knowledge to a file.

        :param filepath: Optional file path for export
        """
        if filepath is None:
            filepath = os.path.join(
                os.path.expanduser('~'), 
                'learned_knowledge.json'
            )
        
        self.knowledge_base.export(filepath)

    def import_knowledge(self, filepath: str = None):
        """
        Import knowledge from a file.

        :param filepath: Optional file path for import
        """
        if filepath is None:
            filepath = os.path.join(
                os.path.expanduser('~'), 
                'learned_knowledge.json'
            )
        
        self.knowledge_base.import_knowledge(filepath)


class SelfLearningAgent:
    def __init__(self):
        # FlashAttention configuration for CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-3-mini-4k-instruct",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16
        )
        
        # Dynamic resource monitor
        self.resource_monitor = ResourceMonitor(
            memory_threshold=0.8,  # 80% RAM usage
            swap_penalty=0.1
        )

    def generate(self, prompt):
        # Context-aware generation budgeting
        with self.resource_monitor:
            return super().generate(prompt)
        
        # Pre-process with active tools
        if math_pattern.match(query):
            return self.active_tools['math'].execute(query)
        if code_pattern.match(query):
            return self.active_tools['code'].interpret(query)
            
        # Fallback to neural generation
        return self.generate(query)