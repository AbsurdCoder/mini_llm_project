"""
Base Tokenizer class that defines the interface for all tokenizers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of text samples to train on
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        pass

    def get_vocab_size(self) -> int:
        """
        Get the actual size of the vocabulary (number of unique tokens).
        
        Returns:
            The number of tokens in the vocabulary.
        """
        return len(self.token_to_id)
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer vocabulary and parameters to a file.
        
        Args:
            path: Path to save the tokenizer
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "tokenizer_type": self.__class__.__name__
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """
        Load a tokenizer from a file.
        
        Args:
            path: Path to load the tokenizer from
            
        Returns:
            Loaded tokenizer instance
        """
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        
        return tokenizer
