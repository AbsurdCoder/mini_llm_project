"""
Enhanced transformer model components with detailed logging.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, Tuple

# Set up logger
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with detailed logging."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, verbose: bool = False):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            verbose: Whether to log detailed information
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.verbose = verbose
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                layer_name: str = "unknown") -> torch.Tensor:
        """
        Forward pass with detailed logging.
        
        Args:
            q: Query tensor [batch_size, seq_len, d_model]
            k: Key tensor [batch_size, seq_len, d_model]
            v: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, 1, seq_len]
            layer_name: Name of the layer for logging
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        if self.verbose:
            logger.debug(f"ATTENTION: {layer_name} - Processing input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if self.verbose:
            logger.debug(f"ATTENTION: {layer_name} - After projection shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Add mask dimension for heads
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            if self.verbose:
                logger.debug(f"ATTENTION: {layer_name} - Applied attention mask with shape {mask.shape}")
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if self.verbose:
            logger.debug(f"ATTENTION: {layer_name} - Attention weights shape: {attn_weights.shape}")
            # Log attention statistics
            avg_attn = attn_weights.mean().item()
            max_attn = attn_weights.max().item()
            logger.debug(f"ATTENTION: {layer_name} - Attention stats: avg={avg_attn:.4f}, max={max_attn:.4f}")
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.output(context)
        
        if self.verbose:
            logger.debug(f"ATTENTION: {layer_name} - Output shape: {output.shape}")
        
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with detailed logging."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, verbose: bool = False):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            verbose: Whether to log detailed information
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose
        
    def forward(self, x: torch.Tensor, layer_name: str = "unknown") -> torch.Tensor:
        """
        Forward pass with detailed logging.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            layer_name: Name of the layer for logging
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        if self.verbose:
            logger.debug(f"FFN: {layer_name} - Processing input shape: {x.shape}")
        
        # First linear layer and activation
        ff_output = F.gelu(self.linear1(x))
        
        if self.verbose:
            logger.debug(f"FFN: {layer_name} - After first linear + GELU shape: {ff_output.shape}")
            # Log activation statistics
            avg_act = ff_output.mean().item()
            max_act = ff_output.max().item()
            logger.debug(f"FFN: {layer_name} - Activation stats: avg={avg_act:.4f}, max={max_act:.4f}")
        
        # Dropout and second linear layer
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)
        
        if self.verbose:
            logger.debug(f"FFN: {layer_name} - Output shape: {ff_output.shape}")
        
        return ff_output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models with detailed logging."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1, verbose: bool = False):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            verbose: Whether to log detailed information
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with detailed logging.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        if self.verbose:
            logger.debug(f"POSITIONAL ENCODING - Processing input shape: {x.shape}")
            logger.debug(f"POSITIONAL ENCODING - Adding positional encodings up to position {x.size(1)}")
        
        x = x + self.pe[:, :x.size(1)]
        output = self.dropout(x)
        
        if self.verbose:
            logger.debug(f"POSITIONAL ENCODING - Output shape: {output.shape}")
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with detailed logging."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, verbose: bool = False, layer_idx: int = 0):
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            verbose: Whether to log detailed information
            layer_idx: Layer index for logging
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.verbose = verbose
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, verbose)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, verbose)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with detailed logging.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, 1, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        layer_name = f"Encoder-Layer-{self.layer_idx}"
        
        if self.verbose:
            logger.debug(f"ENCODER: {layer_name} - Processing input shape: {x.shape}")
        
        # Self-attention with residual connection and layer normalization
        if self.verbose:
            logger.debug(f"ENCODER: {layer_name} - Starting self-attention")
        
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask, layer_name=layer_name)
        
        if self.verbose:
            logger.debug(f"ENCODER: {layer_name} - Applying residual connection and layer norm")
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        if self.verbose:
            logger.debug(f"ENCODER: {layer_name} - Starting feed-forward network")
        
        ff_output = self.feed_forward(x, layer_name=layer_name)
        
        if self.verbose:
            logger.debug(f"ENCODER: {layer_name} - Applying residual connection and layer norm")
        
        x = self.norm2(x + self.dropout(ff_output))
        
        if self.verbose:
            logger.debug(f"ENCODER: {layer_name} - Output shape: {x.shape}")
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with detailed logging."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, verbose: bool = False, layer_idx: int = 0):
        """
        Initialize transformer decoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            verbose: Whether to log detailed information
            layer_idx: Layer index for logging
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.verbose = verbose
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, verbose)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, verbose)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, verbose)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with detailed logging.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            memory: Encoder output tensor [batch_size, src_seq_len, d_model]
            tgt_mask: Optional self-attention mask [batch_size, 1, seq_len]
            memory_mask: Optional cross-attention mask [batch_size, 1, src_seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        layer_name = f"Decoder-Layer-{self.layer_idx}"
        
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Processing input shape: {x.shape}, memory shape: {memory.shape}")
        
        # Self-attention with residual connection and layer normalization
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Starting self-attention")
        
        self_attn_output = self.self_attn(q=x, k=x, v=x, mask=tgt_mask, layer_name=f"{layer_name}-Self")
        
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Applying residual connection and layer norm")
        
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer normalization
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Starting cross-attention with encoder memory")
        
        cross_attn_output = self.cross_attn(q=x, k=memory, v=memory, mask=memory_mask, layer_name=f"{layer_name}-Cross")
        
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Applying residual connection and layer norm")
        
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer normalization
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Starting feed-forward network")
        
        ff_output = self.feed_forward(x, layer_name=layer_name)
        
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Applying residual connection and layer norm")
        
        x = self.norm3(x + self.dropout(ff_output))
        
        if self.verbose:
            logger.debug(f"DECODER: {layer_name} - Output shape: {x.shape}")
        
        return x
