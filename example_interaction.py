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
        """
        Initialize the Autonomous Learning Interface
        """
        # Initialize persistent knowledge agent
        self.persistent_agent = PersistentLearningAgent()

        # Initialize learning agents
        self.self_learning_agent = SelfLearningAgent(autonomous_learning=True)
        self.autonomous_learner = AutonomousLearningAgent(
            learning_interval=600  # 10-minute learning cycle
        )

        # Configure exit handling
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)
        atexit.register(self._cleanup)

        # Configure autonomous learning callback
        def on_learning_complete(learned_knowledge):
            logger.info("Autonomous learning cycle completed")
            logger.info(f"Acquired {len(learned_knowledge)} new knowledge items")
            
            # Save autonomous learning knowledge
            for topic, knowledge in learned_knowledge.items():
                self.persistent_agent.add_knowledge(topic, knowledge)
            
            logger.info("Autonomous knowledge integrated into persistent storage")

        self.autonomous_learner.set_learning_callback(on_learning_complete)
        # Ensure self-learning agent and persistent agent are set up for async operations
        # (if needed, assign them here)

    def _graceful_exit(self, signum=None, frame=None):
        """
        Handle graceful exit with cleanup
        
        :param signum: Signal number
        :param frame: Current stack frame
        """
        print("\n🔄 Initiating graceful shutdown...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """
        Perform cleanup operations
        """
        try:
            # Stop background learning
            if hasattr(self, 'autonomous_learner'):
                self.autonomous_learner.stop_learning()
            
            # Save final knowledge state
            self.persistent_agent.save_knowledge()
            
            print("✅ Cleanup completed successfully.")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def start_background_learning(self):
        """
        Start background autonomous learning
        """
        self.autonomous_learner.start_learning()
        logger.info("Background learning started")

    async def query_agent(self, query: str) -> List[Dict[str, Any]]:
        """
        Asynchronously query the self-learning agent by combining persistent and agent knowledge.
        
        :param query: User query
        :return: List of query results
        """
        try:
            # First, asynchronously search persistent knowledge.
            persistent_results = await self._search_persistent_knowledge(query)
            
            # Wrap the synchronous knowledge base query in a thread
            agent_results = await asyncio.to_thread(self.self_learning_agent.knowledge_base.query, query)
            
            # Combine and return results.
            return persistent_results + agent_results
        except Exception as e:
            logger.error(f"Query error: {e}")
            return [{'error': str(e)}]

    async def _search_persistent_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        Asynchronously search through persistent knowledge.
        
        :param query: Query string to search for.
        :return: List of matching persistent knowledge entries.
        """
        results = []
        # Reload persistent knowledge asynchronously via thread
        self.persistent_agent.knowledge = await asyncio.to_thread(self.persistent_agent._load_knowledge)
        for key, item in self.persistent_agent.knowledge.items():
            content_raw = item.get('content', '')
            if isinstance(content_raw, dict):
                content = (
                    content_raw.get('text') or 
                    content_raw.get('summary') or 
                    content_raw.get('topic') or 
                    str(content_raw)
                )
            else:
                content = str(content_raw)
            if query.lower() in content.lower():
                results.append({
                    'source': 'persistent_knowledge',
                    'key': key,
                    'content': content,
                    'confidence': item.get('confidence', 0.8),
                    'metadata': item.get('metadata', {})
                })
        return results

    async def interact_with_agent(self, interaction: str) -> Dict[str, Any]:
        """
        Asynchronously process an interaction.
        (Adapt further as needed using async wrappers for blocking operations.)
        """
        # Example: wrap the entire processing in a thread if it calls too many sync functions.
        result = await asyncio.to_thread(self._sync_interact, interaction)
        return result

    def _sync_interact(self, interaction: str) -> Dict[str, Any]:
        """
        Synchronous fallback method to process interaction.
        (This can be gradually converted into fully async code.)
        """
        try:
            # Process the interaction
            doc = self.self_learning_agent.nlp(interaction)
            
            # Prepare knowledge entry
            knowledge_entry = {
                'content': interaction,
                'entities': [{'text': ent.text, 'label': ent.label_} for ent in doc.ents],
                'source': 'user_interaction',
                'confidence': 0.8,
                'metadata': {'source': 'user_interaction'}
            }
            
            # Store in persistent knowledge
            key = f"interaction_{int(time.time())}"
            self.persistent_agent.add_knowledge(key, knowledge_entry)
            
            return {
                'status': 'success',
                'message': f"Learned: {interaction}",
                'knowledge': knowledge_entry
            }
            
        except Exception as e:
            logger.error(f"Interaction error: {e}")
            return {'error': str(e)}

    def conversational_teach(self, statement: str):
        """
        Enable direct conversational knowledge teaching.
        
        :param statement: Knowledge statement to learn
        """
        try:
            # Extract key information using NLP
            doc = self.self_learning_agent.nlp(statement)
            
            # Identify entities and their types
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
            
            # Prepare knowledge entry
            knowledge_entry = {
                'content': statement,
                'entities': entities,
                'source': 'conversation',
                'confidence': 0.8,
                'metadata': {'source': 'conversation'}
            }
            
            # Validate and store knowledge
            if self.self_learning_agent.autonomous_learner._validate_knowledge(knowledge_entry):
                key = f"conversation_{int(time.time())}"
                self.self_learning_agent.autonomous_learner._persist_learned_knowledge(knowledge_entry)
                print(f"🧠 Learned: {statement}")
            else:
                print("🤔 Similar knowledge already exists.")
        
        except Exception as e:
            print(f"❌ Learning error: {e}")

    def interactive_cli(self):
        """
        Enhanced interactive CLI with conversational learning.
        """
        print("🌟 Conversational AI Learning Interface 🧠")
        print("Commands: learn, query, teach, context, exit")
        
        while True:
            try:
                user_input = input("\n🗣️ You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'learn':
                    self.self_learning_agent.learn()
                elif user_input.lower() == 'query':
                    query = input("📝 Enter query: ")
                    results = self.self_learning_agent.knowledge_base.query(query)
                    self.display_query_results(results)
                elif user_input.lower() == 'teach':
                    statement = input("📚 Enter knowledge to teach: ")
                    self.conversational_teach(statement)
                elif user_input.lower() == 'context':
                    context = input("🌐 Set conversational context: ")
                    print(f"Context set: {context}")
                else:
                    if len(user_input.split()) > 3:
                        self.conversational_teach(user_input)
                    else:
                        results = self.self_learning_agent.knowledge_base.query(user_input)
                        self.display_query_results(results)
            
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def display_query_results(self, results: List[Dict[str, Any]]):
        """
        Display query results with improved mathematical result handling.
        """
        print("\n🔍 Query Results:")
        if not results:
            print("No matching results found.")
            return

        for result in results:
            # Handle mathematical computation results
            if result.get('metadata', {}).get('type') == 'mathematical_computation':
                print(f"📐 Maths results {result['content']}")
                print(f"confidence: {result.get('confidence', 1.0):.2f}")
                continue

            # Handle other types of results
            content = result.get('content', '')
            confidence = result.get('confidence', 0.0)
            
            if isinstance(content, dict):
                content = content.get('text', str(content))
            
            print(f"\nConfidence: {confidence:.2f}")
            print(f"Content: {content}")

            # Display metadata if available
            metadata = result.get('metadata', {})
            if metadata:
                print("Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")

    async def process_command(self, command: str):
        """Process user commands with improved result display."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "query":
            results = await self.query_agent(arg)
            self.display_query_results(results)
        # ... rest of command processing ...

    async def run_interactive_cli(self):
        while True:
            try:
                user_input = input("\n🌟 Enter command: ").strip()
                if user_input.lower().startswith('query'):
                    await self.process_command(user_input)  # Properly await the coroutine
                elif user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'learn':
                    self.start_background_learning()
                    print("🎓 Started background learning")
                elif user_input.lower().startswith('interact'):
                    message = user_input[8:].strip()
                    if message:
                        result = await self.interact_with_agent(message)
                        print(f"🤖 {result.get('message', 'Processed interaction')}")
                    else:
                        print("❌ Please provide a message to interact with")
                elif user_input.lower().startswith('explore'):
                    domain = user_input[8:].strip()
                    if domain:
                        print(f"🌍 Exploring domain: {domain}")
                        insights = self.autonomous_learner._generate_synthetic_knowledge()
                        print("\n🔍 Generated Insights:")
                        for key, insight in insights.items():
                            print(f"- {insight['text']} (Confidence: {insight.get('confidence', 0.5)})")
                elif user_input.lower() == 'show knowledge':
                    print("📖 Current Knowledge Base:")
                    knowledge = self.self_learning_agent.knowledge_base.knowledge
                    for key, entry in knowledge.items():
                        print(f"🔑 {key}: {entry.get('content', 'No details')}")
                elif user_input.lower() == 'help':
                    print("""
                          🤖 Autonomous Learning AI Interface 🧠
    Commands:
      'learn': Start autonomous background learning
      'query [topic]': Search knowledge base
      'interact [message]': Teach the AI
      'explore [domain]': Generate insights about a topic
      'show knowledge': Display current knowledge
      'exit': Quit the interface
      'help': Show this help message
                    """)
                else:
                    print("❓ Unknown command. Type 'help' for available commands")
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """
    Main entry point for the Autonomous Learning Interface
    """
    print("🤖 Autonomous Learning AI Interface 🧠")
    print("Type 'help' for commands, 'exit' to quit\n")
    
    interface = AutonomousLearningInterface()
    asyncio.run(interface.run_interactive_cli())

if __name__ == "__main__":
    main()