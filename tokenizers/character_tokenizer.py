"""
Character-level tokenizer implementation.
"""
from typing import List, Dict, Set
import collections
from .base_tokenizer import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """
    A simple character-level tokenizer that treats each character as a token.
    Special tokens like [PAD], [UNK], [BOS], [EOS] are also included.
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize the character tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (usually not reached for character tokenizer)
        """
        super().__init__(vocab_size)
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }
        
        # Initialize with special tokens
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
    def train(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text samples to train on
        """
        # Count character frequencies
        char_counts = collections.Counter()
        for text in texts:
            char_counts.update(text)
        
        # Sort by frequency
        sorted_chars = sorted(
            char_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Add characters to vocabulary (up to vocab_size)
        next_id = len(self.special_tokens)
        for char, _ in sorted_chars:
            if next_id >= self.vocab_size:
                break
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        return [self.token_to_id.get(char, self.special_tokens["[UNK]"]) for char in text]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return "".join([self.id_to_token.get(id, "") for id in ids 
                       if id not in [self.special_tokens["[PAD]"], 
                                    self.special_tokens["[BOS]"], 
                                    self.special_tokens["[EOS]"]]])
    
    def add_special_tokens(self, text: str) -> List[int]:
        """
        Add special tokens (BOS/EOS) to the encoded text.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs with special tokens
        """
        encoded = self.encode(text)
        return [self.special_tokens["[BOS]"]] + encoded + [self.special_tokens["[EOS]"]]
