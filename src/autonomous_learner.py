import os
import re
import json
import logging
import threading
import time
import random
import asyncio
from typing import Dict, Any, Optional, Callable
from bs4 import BeautifulSoup

import wikipedia
import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RateLimitedAPIClient:
    """
    A rate-limited API client to manage request frequency and prevent overloading.
    """
    def __init__(self, max_requests_per_minute=10):
        """
        Initialize the rate-limited API client.
        
        :param max_requests_per_minute: Maximum number of requests allowed per minute
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.request_timestamps = []

    def _clean_old_timestamps(self):
        """Remove timestamps older than one minute."""
        current_time = time.time()
        self.request_timestamps = [
            timestamp for timestamp in self.request_timestamps 
            if current_time - timestamp < 60
        ]

    def can_make_request(self) -> bool:
        """
        Check if a request can be made based on rate limiting.
        
        :return: Boolean indicating if a request is allowed
        """
        self._clean_old_timestamps()
        return len(self.request_timestamps) < self.max_requests_per_minute

    def record_request(self):
        """Record the timestamp of a request."""
        self.request_timestamps.append(time.time())

class AutonomousLearningAgent:
    def __init__(self, learning_interval=300, max_requests_per_minute=10):
        """
        Initialize the Autonomous Learning Agent with enhanced knowledge management.
        
        :param learning_interval: Interval between learning cycles in seconds
        :param max_requests_per_minute: Maximum API requests per minute
        """
        # Learning configuration
        self.learning_interval = learning_interval
        self.learning_thread = None
        self.is_learning = False
        self.learning_callback = None
        
        # Rate limiting
        self.api_client = RateLimitedAPIClient(max_requests_per_minute)
        
        # NLP and text processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("SpaCy model not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        
        self.vectorizer = TfidfVectorizer()
        
        # Initialize knowledge base if not already initialized
        if not hasattr(self, 'knowledge_base'):
            from .knowledge_base import KnowledgeBase
            self.knowledge_base = KnowledgeBase()
        
        # Logging setup
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Learning sources with prioritization
        self.learning_sources = [
            (self._learn_from_wikipedia, 0.7),     # High priority
            (self._learn_from_scientific_sources, 0.8),  # Highest priority
            (self._learn_from_random_articles, 0.5),    # Medium priority
            (self._generate_synthetic_knowledge, 0.3),   # Low priority
            (self._learn_from_web_sources, 0.4)  # New learning source
        ]

    def set_learning_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Set a callback function to be called after learning completes.
        
        :param callback: Function to call with learned knowledge
        """
        self.learning_callback = callback

    def _validate_knowledge(self, new_knowledge: Dict[str, Any], existing_knowledge: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate whether new knowledge is sufficiently different from existing knowledge.
        Uses cosine similarity on text content to detect redundancy.
        
        :param new_knowledge: Newly learned item containing 'content'
        :param existing_knowledge: Dictionary of existing knowledge entries.
        :return: True if the new knowledge is valid (not a duplicate), False otherwise.
        """
        new_content = new_knowledge.get('content', '')
        if not new_content:
            self.logger.warning("New knowledge has no content.")
            return False

        # If existing knowledge is provided, compare using cosine similarity.
        if existing_knowledge:
            for key, existing_item in existing_knowledge.items():
                existing_content = existing_item.get('content', '')
                if not existing_content:
                    continue

                # Vectorize both new and existing content.
                corpus = [new_content, existing_content]
                tfidf_matrix = self.vectorizer.fit_transform(corpus)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
                # Threshold can be adjusted as needed; here we use 0.9 for high similarity.
                if similarity > 0.90:
                    self.logger.info(f"Duplicate detected for key '{key}' with similarity: {similarity:.2f}")
                    return False
        return True

    def _persist_learned_knowledge(self, learned_item):
        """
        Persist learned knowledge to the knowledge base and ensure it's saved.
        
        :param learned_item: Dictionary containing learned information
        :return: Key of persisted knowledge or None
        """
        try:
            # Generate a unique key for the learned item
            key = f"{learned_item.get('topic', 'unknown')}_{int(time.time())}"
            
            # Prepare knowledge entry
            knowledge_entry = {
                'content': learned_item,
                'timestamp': time.time(),
                'source': 'autonomous_learning'
            }
            
            # Store in knowledge base
            self.knowledge_base.store(key, knowledge_entry)
            
            # Save to persistent storage
            self.knowledge_base.save_ephemeral()
            self.knowledge_base.save_persistent()
            
            self.logger.info(f"Persisted knowledge: {key}")
            return key
        
        except Exception as e:
            self.logger.error(f"Failed to persist knowledge: {e}")
            return None

    def _learn_from_wikipedia(self) -> Dict[str, Any]:
        """
        Learn from a random Wikipedia article.
        
        :return: Dictionary of learned knowledge
        """
        learned_knowledge = {}
        
        if not self.api_client.can_make_request():
            self.logger.warning("Rate limit reached. Skipping Wikipedia learning.")
            return learned_knowledge
        
        try:
            # Fetch a random Wikipedia article
            random_topic = wikipedia.random(pages=1)
            self.logger.info(f"Learning about: {random_topic}")
            
            page = wikipedia.page(random_topic)
            summary = page.summary
            
            # Process text with SpaCy
            doc = self.nlp(summary)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            learned_knowledge[random_topic] = {
                'summary': summary,
                'source': 'Wikipedia',
                'url': page.url,
                'topic': random_topic,
                'entities': entities
            }
            
            self.api_client.record_request()
            self.logger.info(f"Learned about {random_topic}")
            
        except wikipedia.exceptions.DisambiguationError as e:
            self.logger.warning(f"Disambiguation error for {random_topic}: {e}")
        except wikipedia.exceptions.PageError:
            self.logger.warning(f"Page not found for {random_topic}")
        except Exception as e:
            self.logger.error(f"Wikipedia learning error: {e}")
        
        return learned_knowledge

    def _learn_from_random_articles(self) -> Dict[str, Any]:
        """
        Learn from curated random articles or web sources.
        
        :return: Dictionary of learned knowledge
        """
        learned_knowledge = {}
        
        if not self.api_client.can_make_request():
            self.logger.warning("Rate limit reached. Skipping random article learning.")
            return learned_knowledge
        
        # List of potential learning sources
        sources = [
            "https://en.wikipedia.org/wiki/Special:Random",
            "https://www.gutenberg.org/browse/scores/top",
            "https://www.science.org/topic/article/random"
        ]
        
        try:
            source_url = random.choice(sources)
            response = requests.get(source_url, timeout=10)
            
            if response.status_code == 200:
                # Basic text extraction (can be improved with more sophisticated parsing)
                text = re.sub(r'\s+', ' ', response.text[:1000])
                
                doc = self.nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                
                learned_knowledge[source_url] = {
                    'summary': text,
                    'source': 'Random Web Article',
                    'url': source_url,
                    'entities': entities
                }
                
                self.api_client.record_request()
                self.logger.info(f"Learned from {source_url}")
            
        except Exception as e:
            self.logger.error(f"Random article learning error: {e}")
        
        return learned_knowledge

    def _learn_from_scientific_sources(self) -> Dict[str, Any]:
        """
        Learn from scientific and educational sources.
        
        :return: Dictionary of learned knowledge
        """
        scientific_sources = [
            "https://arxiv.org/list/cs.AI/recent",  # AI research papers
            "https://www.nature.com/subjects/artificial-intelligence",
            "https://www.science.org/topic/article-type/research-article"
        ]
        
        learned_knowledge = {}
        
        for source in scientific_sources:
            try:
                if not self.api_client.can_make_request():
                    break
                
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    # Extract meaningful text (simplified example)
                    doc = self.nlp(response.text[:5000])  # Process first 5000 chars
                    
                    # Extract key sentences and concepts
                    key_sentences = [
                        sent.text for sent in doc.sents 
                        if len(sent.text.split()) > 5  # Meaningful sentences
                    ]
                    
                    for sentence in key_sentences:
                        # Generate a unique key
                        key = f"sci_knowledge_{hash(sentence)}"
                        
                        # Validate and store knowledge
                        knowledge_entry = {
                            'text': sentence,
                            'source': source,
                            'confidence': 0.7  # Scientific sources have higher confidence
                        }
                        
                        if self._validate_knowledge(knowledge_entry):
                            learned_knowledge[key] = knowledge_entry
                
                self.api_client.record_request()
            except Exception as e:
                self.logger.error(f"Error learning from {source}: {e}")
        
        return learned_knowledge

    def _generate_synthetic_knowledge(self) -> Dict[str, Any]:
        """
        Generate synthetic knowledge by combining existing concepts.
        
        :return: Dictionary of generated knowledge
        """
        synthetic_knowledge = {}
        
        # Knowledge generation templates
        templates = [
            "The intersection of {domain1} and {domain2} reveals {insight}",
            "{concept} can be understood through the lens of {perspective}",
            "Emerging trends in {field} suggest {prediction}"
        ]
        
        domains = [
            "artificial intelligence", "machine learning", "neuroscience", 
            "cognitive psychology", "data science", "robotics"
        ]
        
        for _ in range(5):  # Generate 5 synthetic knowledge entries
            domain1, domain2 = random.sample(domains, 2)
            template = random.choice(templates)
            
            synthetic_text = template.format(
                domain1=domain1, 
                domain2=domain2, 
                concept=domain1,
                perspective=domain2,
                field=domain1,
                insight=f"new possibilities in {domain2}",
                prediction="transformative innovations"
            )
            
            key = f"synthetic_knowledge_{hash(synthetic_text)}"
            synthetic_knowledge[key] = {
                'text': synthetic_text,
                'source': 'synthetic_generation',
                'confidence': 0.4  # Lower confidence for synthetic knowledge
            }
        
        return synthetic_knowledge

    def _learn_from_web_sources(self) -> Dict[str, Any]:
        """
        Learn from diverse web sources with robust knowledge extraction.
        
        :return: Dictionary of learned knowledge
        """
        learning_sources = [
            {
                'url': 'https://en.wikipedia.org/wiki/Special:Random',
                'extractor': self._extract_wikipedia_knowledge
            },
            {
                'url': 'https://www.britannica.com/browse/Science-and-Technology',
                'extractor': self._extract_britannica_knowledge
            }
        ]
        
        learned_knowledge = {}
        
        for source in learning_sources:
            try:
                # Respect API request limits
                if not self.api_client.can_make_request():
                    break
                
                response = requests.get(source['url'], timeout=10)
                
                if response.status_code == 200:
                    # Extract knowledge using source-specific extractor
                    extracted_knowledge = source['extractor'](response.text)
                    
                    # Validate and store each piece of knowledge
                    for topic, details in extracted_knowledge.items():
                        if self._validate_knowledge(details):
                            key = f"{topic}_{int(time.time())}"
                            learned_knowledge[key] = {
                                'topic': topic,
                                'details': details,
                                'source': source['url']
                            }
                            
                            # Persist immediately
                            self._persist_learned_knowledge(learned_knowledge[key])
                    
                self.api_client.record_request()
        
            except Exception as e:
                self.logger.error(f"Error learning from {source['url']}: {e}")
        
        return learned_knowledge

    def learn(self) -> Dict[str, Any]:
        """
        Comprehensive learning method combining multiple sources.
        
        :return: Dictionary of learned knowledge
        """
        self.logger.info('Starting learning process...')
        learned_knowledge = {}
        
        for method, priority in self.learning_sources:
            try:
                method_knowledge = method()
                
                # Integrate knowledge with priority-based filtering
                for key, knowledge in method_knowledge.items():
                    if self._validate_knowledge(knowledge):
                        if random.random() < priority:
                            learned_key = self._persist_learned_knowledge(knowledge)
                            if learned_key:
                                learned_knowledge[learned_key] = knowledge
            
            except Exception as e:
                self.logger.error(f"Learning method {method.__name__} failed: {e}")
        
        self.logger.info('Learning process completed.')
        return learned_knowledge

    def start_learning(self):
        """
        Start background autonomous learning process using an asynchronous task.
        If no running event loop exists (e.g. in a synchronous context), create one.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if not self.is_learning:
            self.is_learning = True
            self._learning_task = loop.create_task(self._async_learning_loop())
            self.logger.info("Autonomous learning started (async).")

    def stop_learning(self):
        """
        Stop the autonomous learning process.
        """
        self.is_learning = False
        if hasattr(self, '_learning_task'):
            self._learning_task.cancel()
        self.logger.info("Autonomous learning stopped")

    async def _async_learning_loop(self):
        """
        Background asynchronous learning loop.
        """
        while self.is_learning:
            try:
                # Wrap synchronous learning method in a thread if needed
                learned_knowledge = await asyncio.to_thread(self.learn)
                if self.learning_callback:
                    # Optionally, run callback in thread if it performs blocking operations
                    await asyncio.to_thread(self.learning_callback, learned_knowledge)
            except Exception as e:
                self.logger.error(f"Learning loop error: {e}")
            await asyncio.sleep(self.learning_interval)

    def export_knowledge(self, filepath: Optional[str] = None) -> bool:
        """
        Export learned knowledge to a JSON file.
        
        :param filepath: Path to export knowledge
        :return: Export success status
        """
        if filepath is None:
            filepath = os.path.join(
                os.path.expanduser('~'), 
                'autonomous_learning_export.json'
            )
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            self.logger.info(f"Knowledge exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Knowledge export error: {e}")
            return False

    def _extract_wikipedia_knowledge(self, html_content):
        """
        Extract structured knowledge from Wikipedia page.
        
        :param html_content: HTML content of the page
        :return: Dictionary of extracted knowledge
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title and first paragraph
            title = soup.find('h1', {'id': 'firstHeading'}).text.strip()
            first_paragraph = soup.find('div', {'class': 'mw-parser-output'}).find('p').text.strip()
            
            # Use NLP to extract key entities and concepts
            doc = self.nlp(first_paragraph)
            
            entities = [
                {
                    'text': ent.text, 
                    'label': ent.label_
                } for ent in doc.ents
            ]
            
            return {
                title: {
                    'summary': first_paragraph,
                    'entities': entities,
                    'source': 'Wikipedia',
                    'confidence': 0.6
                }
            }
        
        except Exception as e:
            self.logger.warning(f"Wikipedia knowledge extraction failed: {e}")
            return {}

    def _extract_britannica_knowledge(self, html_content):
        """
        Extract structured knowledge from Britannica page.
        
        :param html_content: HTML content of the page
        :return: Dictionary of extracted knowledge
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title and summary
            title = soup.find('h1', {'class': 'article-title'}).text.strip()
            summary = soup.find('div', {'class': 'article-body'}).find('p').text.strip()
            
            # Use NLP for entity extraction
            doc = self.nlp(summary)
            
            entities = [
                {
                    'text': ent.text, 
                    'label': ent.label_
                } for ent in doc.ents
            ]
            
            return {
                title: {
                    'summary': summary,
                    'entities': entities,
                    'source': 'Britannica',
                    'confidence': 0.7
                }
            }
        
        except Exception as e:
            self.logger.warning(f"Britannica knowledge extraction failed: {e}")
            return {}