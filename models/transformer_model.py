"""
Transformer model implementation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .base_model import BaseModel
from .transformer_components import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)


class TransformerModel(BaseModel):
    """
    A small transformer model for language modeling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer model.
        
        Args:
            config: Model configuration parameters
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.vocab_size = config.get("vocab_size", 10000)
        self.d_model = config.get("d_model", 256)
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 4)
        
        # Ensure d_ff is a multiple of d_model (typically 4x)
        # This helps prevent dimension mismatches
        self.d_ff = config.get("d_ff", self.d_model * 4)
        
        self.max_seq_len = config.get("max_seq_len", 512)
        self.dropout = config.get("dropout", 0.1)
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens [batch_size, seq_len]
            
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        # Create attention mask for transformer if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert attention mask to format expected by transformer
        # [batch_size, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1)
        
        # Embed tokens and add positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Apply transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=attention_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: Number of highest probability tokens to keep for sampling
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize sequence with input_ids
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get model predictions
                logits = self(generated)
                
                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Convert logits to probabilities
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from the probability distribution
                next_token_idx = torch.multinomial(probs, num_samples=1)
                
                # Convert top-k indices to actual token indices
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class DecoderOnlyTransformer(BaseModel):
    """
    A decoder-only transformer model (similar to GPT architecture).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the decoder-only transformer model.
        
        Args:
            config: Model configuration parameters
        """
        super().__init__(config)
        
        # Extract configuration parameters
        self.vocab_size = config.get("vocab_size", 10000)
        self.d_model = config.get("d_model", 256)
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 4)
        
        # Ensure d_ff is a multiple of d_model (typically 4x)
        # This helps prevent dimension mismatches
        self.d_ff = config.get("d_ff", self.d_model * 4)
        
        self.max_seq_len = config.get("max_seq_len", 512)
        self.dropout = config.get("dropout", 0.1)
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        )
        
        # Transformer decoder layers (using encoder layers with causal mask)
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal mask for autoregressive decoding.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [seq_len, seq_len]
        """
        # Create lower triangular mask (1s in lower triangle, 0s elsewhere)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens [batch_size, seq_len]
            
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create attention mask for transformer if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Embed tokens and add positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Apply transformer decoder layers
        # For each layer, we need to create the appropriate mask format
        for decoder_layer in self.decoder_layers:
            # Create a mask that combines the attention mask and causal mask
            # First, reshape attention_mask to [batch_size, 1, seq_len]
            layer_mask = attention_mask.unsqueeze(1)
            
            # Apply the decoder layer with the properly formatted mask
            x = decoder_layer(x, mask=layer_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: Number of highest probability tokens to keep for sampling
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize sequence with input_ids
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get model predictions
                logits = self(generated)
                
                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Convert logits to probabilities
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from the probability distribution
                next_token_idx = torch.multinomial(probs, num_samples=1)
                
                # Convert top-k indices to actual token indices
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
