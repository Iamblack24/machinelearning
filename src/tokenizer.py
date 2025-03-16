from typing import List, Dict
import re
from collections import Counter

class CustomTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
    def train(self, texts: List[str]):
        """Train tokenizer on a corpus of texts"""
        # Basic word-level tokenization
        words = []
        for text in texts:
            words.extend(self._preprocess(text))
            
        # Build vocabulary based on frequency
        counter = Counter(words)
        vocab_words = [word for word, _ in counter.most_common(self.vocab_size - len(self.special_tokens))]
        
        # Create vocabulary mappings
        self.vocab = {**self.special_tokens}
        for i, word in enumerate(vocab_words, len(self.special_tokens)):
            self.vocab[word] = i
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _preprocess(self, text: str) -> List[str]:
        """Preprocess and tokenize text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()