"""
Byte Pair Encoding (BPE) tokenizer implementation with improved spacing handling.
"""
from typing import List, Dict, Tuple, Counter
import re
import collections
from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    A Byte Pair Encoding (BPE) tokenizer that learns subword units from text.
    """
    
    def __init__(self, vocab_size: int = 10000, min_frequency: int = 2):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for a token to be considered
        """
        super().__init__(vocab_size)
        self.min_frequency = min_frequency
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }
        
        # Initialize with special tokens
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # BPE specific attributes
        self.merges = {}  # Dictionary of merge operations
        self.word_vocab = {}  # Dictionary of words and their frequencies
        self.pattern = None  # Regex pattern for tokenization
        
        # Track which tokens should be followed by a space
        self.token_boundaries = set()
        
    def _get_stats(self, vocab: Dict[str, int]) -> Counter:
        """
        Count frequency of symbol pairs in the vocabulary.
        
        Args:
            vocab: Dictionary of words and their frequencies
            
        Returns:
            Counter of symbol pairs
        """
        pairs = collections.Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """
        Merge a pair of symbols in the vocabulary.
        
        Args:
            pair: Pair of symbols to merge
            vocab: Dictionary of words and their frequencies
            
        Returns:
            Updated vocabulary
        """
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_vocab = {}
        
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
            
        return new_vocab
    
    def train(self, texts: List[str]) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of text samples to train on
        """
        # Initialize vocabulary with character-level tokens
        word_vocab = {}
        
        # Track word boundaries for better space handling
        word_boundaries = set()
        
        for text in texts:
            # Split text into words
            words = re.findall(r'\w+|[^\w\s]', text)
            for word in words:
                # Add space between characters
                char_word = ' '.join(list(word))
                if char_word not in word_vocab:
                    word_vocab[char_word] = 0
                word_vocab[char_word] += 1
                
                # Mark the last character of each word for space handling
                word_boundaries.add(word[-1])
        
        # Filter by frequency
        word_vocab = {k: v for k, v in word_vocab.items() if v >= self.min_frequency}
        self.word_vocab = word_vocab
        
        # Initialize character vocabulary
        chars = set()
        for word in word_vocab.keys():
            for char in word.split():
                chars.add(char)
        
        # Add characters to vocabulary
        next_id = len(self.special_tokens)
        for char in sorted(chars):
            if next_id >= self.vocab_size:
                break
            self.token_to_id[char] = next_id
            self.id_to_token[next_id] = char
            
            # Mark tokens that are word boundaries
            if char in word_boundaries:
                self.token_boundaries.add(char)
                
            next_id += 1
        
        # Learn BPE merges
        num_merges = min(self.vocab_size - next_id, 20000)  # Limit number of merges
        
        for i in range(num_merges):
            if len(self.token_to_id) >= self.vocab_size:
                break
                
            # Get pair statistics
            pairs = self._get_stats(word_vocab)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_vocab = self._merge_vocab(best_pair, word_vocab)
            
            # Add the merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = next_id
                self.id_to_token[next_id] = merged_token
                
                # Check if the merged token ends with a word boundary character
                if any(merged_token.endswith(boundary) for boundary in word_boundaries):
                    self.token_boundaries.add(merged_token)
                    
                next_id += 1
            
            # Record the merge operation
            self.merges[best_pair] = merged_token
        
        # Create regex pattern for tokenization
        self.pattern = re.compile(r'|'.join(re.escape(token) for token in sorted(
            self.token_to_id.keys(), key=len, reverse=True) if token not in self.special_tokens.keys()))
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not self.pattern:
            return list(text)  # Fallback to character tokenization
        
        tokens = []
        # Split text into words and spaces
        for match in re.finditer(r'(\s+|\w+|[^\w\s])', text):
            span = match.group(0)
            
            # Keep spaces as separate tokens
            if span.isspace():
                tokens.append(" ")
                continue
                
            # Process non-space spans
            if not span:
                continue
                
            # Find all matches
            matches = self.pattern.findall(span)
            if matches:
                tokens.extend(matches)
            else:
                # If no matches, split into characters
                tokens.extend(list(span))
                
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        return [self.token_to_id.get(token, self.special_tokens["[UNK]"]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text with proper spacing.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text with proper spacing
        """
        # Filter out special tokens
        filtered_ids = [id for id in ids 
                      if id not in [self.special_tokens["[PAD]"], 
                                   self.special_tokens["[BOS]"], 
                                   self.special_tokens["[EOS]"]]]
        
        # Convert IDs to tokens
        tokens = [self.id_to_token.get(id, "[UNK]") for id in filtered_ids]
        
        # Process tokens with proper spacing
        result = ""
        for i, token in enumerate(tokens):
            # Skip explicit space tokens
            if token == " ":
                result += " "
                continue
                
            # Add the token
            result += token
            
            # Add space after tokens that should be followed by space
            # But only if not at the end and the next token isn't a space
            if (token in self.token_boundaries and 
                i < len(tokens) - 1 and 
                tokens[i+1] != " "):
                result += " "
                
            # Add space after punctuation that should be followed by space
            if (token in ".,!?;:" and 
                i < len(tokens) - 1 and 
                tokens[i+1] != " "):
                result += " "
        
        # Post-process to fix common spacing issues
        result = re.sub(r'\s+', ' ', result)  # Replace multiple spaces with single space
        result = re.sub(r'\s([.,!?;:])', r'\1', result)  # Remove space before punctuation
        result = re.sub(r'([.,!?;:])\s', r'\1 ', result)  # Ensure space after punctuation
        
        return result
    
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
        
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file.
        
        Args:
            path: Path to save the tokenizer
        """
        data = {
            "token_to_id": self.token_to_id,
            "merges": self.merges,
            "token_boundaries": list(self.token_boundaries),
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency
        }
        super()._save(path, data)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """
        Load a tokenizer from a file.
        
        Args:
            path: Path to load the tokenizer from
            
        Returns:
            Loaded tokenizer
        """
        data = cls._load(path)
        tokenizer = cls(
            vocab_size=data.get("vocab_size", 10000),
            min_frequency=data.get("min_frequency", 2)
        )
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(id): token for token, id in data["token_to_id"].items()}
        tokenizer.merges = data.get("merges", {})
        tokenizer.token_boundaries = set(data.get("token_boundaries", []))
        
        # Recreate pattern for tokenization
        tokenizer.pattern = re.compile(r'|'.join(re.escape(token) for token in sorted(
            tokenizer.token_to_id.keys(), key=len, reverse=True) if token not in tokenizer.special_tokens.keys()))
        
        return tokenizer
