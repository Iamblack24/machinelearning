import os
import sys
import json
import time
import threading
import signal
import atexit
import asyncio  # <-- Added for asynchronous processing
from typing import List, Dict, Any

from src.learner import SelfLearningAgent
from src.autonomous_learner import AutonomousLearningAgent
from loguru import logger

# Loguru comes with a default configuration.
# Optionally, you can add file logging:
logger.add("autonomous_learning.log", rotation="1 MB", level="INFO")

class PersistentLearningAgent:
    def __init__(self, knowledge_file: str = None):
        # Default persistent file in the project root (two levels up from src)
        if knowledge_file is None:
            knowledge_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'persistent_knowledge.json')
        self.knowledge_file = os.path.abspath(knowledge_file)
        self.knowledge = self._load_knowledge()

    def _load_knowledge(self) -> Dict[str, Any]:
        """
        Load existing knowledge from file.
        """
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")
            return {}

    def save_knowledge(self):
        """
        Save current knowledge to persistent storage.
        """
        try:
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge, f, indent=2)
            logger.info(f"Knowledge saved to {self.knowledge_file}")
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")

    def add_knowledge(self, key: str, value: Any):
        """
        Add new knowledge to the persistent store by merging with existing entries.
        """
        # Refresh current knowledge from disk.
        current_knowledge = self._load_knowledge()
        unique_key = f"{key}_{int(time.time())}"
        
        if isinstance(value, dict):
            content = value.get("content", str(value))
            confidence = value.get("confidence", 0.8)
            metadata = value.get("metadata", {"source": value.get("source", "user_interaction")})
        else:
            content = str(value)
            confidence = 0.8
            metadata = {"source": "user_interaction"}
            
        current_knowledge[unique_key] = {
            'content': content,
            'timestamp': time.time(),
            'confidence': confidence,
            'metadata': metadata
        }
        self.knowledge = current_knowledge
        self.save_knowledge()

class AutonomousLearningInterface:
    def __init__(self):
        """Initialize the Autonomous Learning Interface with custom foundational model"""
        self.persistent_agent = PersistentLearningAgent()
        
        # Enhanced training configuration
        self.training_config = {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 1e-4,
            'warmup_steps': 4000,
            'conversation_style': {
                'humor_level': 0.7,
                'formality': 0.5,
                'creativity': 0.8
            },
            'context_window': 1024,
            'attention_layers': 12
        }
        
        # Initialize specialized processors
        self.conversation_processor = ConversationProcessor(
            style_config=self.training_config['conversation_style']
        )
        self.context_manager = ContextManager(
            window_size=self.training_config['context_window']
        )
        
        # Initialize learning agents with enhanced capabilities
        self.self_learning_agent = SelfLearningAgent(
            vocab_size=50000,
            d_model=512,
            conversation_processor=self.conversation_processor,
            context_manager=self.context_manager
        )

    async def train_model(self, training_data: List[str]):
        """Enhanced training with conversation and humor understanding"""
        # Prepare specialized training datasets
        conversation_pairs = self._prepare_conversation_pairs(training_data)
        humor_examples = self._extract_humor_patterns(training_data)
        
        # Multi-task training
        tasks = [
            self._train_conversation(conversation_pairs),
            self._train_humor_understanding(humor_examples),
            self._train_context_awareness(training_data)
        ]
        
        await asyncio.gather(*tasks)

    async def _train_conversation(self, conversation_pairs: List[Tuple[str, str]]):
        """Train conversation capabilities"""
        for epoch in range(self.training_config['epochs']):
            for context, response in conversation_pairs:
                # Train response generation
                loss = await self.self_learning_agent.train_conversation(
                    context, 
                    response,
                    style_config=self.training_config['conversation_style']
                )
                logger.info(f"Conversation Training - Epoch {epoch}, Loss: {loss:.4f}")

    async def _train_humor_understanding(self, humor_examples: List[Dict[str, Any]]):
        """Train humor recognition and generation"""
        for epoch in range(self.training_config['epochs']):
            for example in humor_examples:
                # Train humor patterns
                loss = await self.self_learning_agent.train_humor(
                    example['setup'],
                    example['punchline'],
                    example['humor_type']
                )
                logger.info(f"Humor Training - Epoch {epoch}, Loss: {loss:.4f}")

    def _prepare_conversation_pairs(self, data: List[str]) -> List[Tuple[str, str]]:
        """Prepare conversation pairs for training"""
        pairs = []
        for i in range(0, len(data)-1, 2):
            context = data[i]
            response = data[i+1]
            pairs.append((context, response))
        return pairs

    def _extract_humor_patterns(self, data: List[str]) -> List[Dict[str, Any]]:
        """Extract humor patterns from training data"""
        patterns = []
        for text in data:
            if self.conversation_processor.is_humorous(text):
                humor_components = self.conversation_processor.analyze_humor(text)
                patterns.append(humor_components)
        return patterns

    async def interact_with_agent(self, interaction: str) -> Dict[str, Any]:
        """Enhanced interaction with humor and personality"""
        try:
            # Process interaction with context
            context = self.context_manager.get_current_context()
            
            # Analyze interaction style
            style_analysis = self.conversation_processor.analyze_style(interaction)
            
            # Generate appropriate response
            response = await self.self_learning_agent.generate_response(
                interaction,
                context=context,
                style=style_analysis
            )
            
            # Add humor if appropriate
            if self.conversation_processor.should_add_humor(interaction, response):
                response = self.conversation_processor.enhance_with_humor(response)
            
            # Update context
            self.context_manager.update_context(interaction, response)
            
            return {
                'status': 'success',
                'message': response,
                'style': style_analysis,
                'humor_level': self.conversation_processor.measure_humor(response)
            }
            
        except Exception as e:
            logger.error(f"Interaction error: {e}")
            return {'error': str(e)}

    # ... rest of the existing code ...