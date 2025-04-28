"""
Encoder-only transformer model implementation (BERT-like).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base_model import BaseModel
from .transformer_components import (
    PositionalEncoding,
    TransformerEncoderLayer
)

class BertMLMHead(nn.Module):
    """Prediction head for Masked Language Modeling (MLM)."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config.get("d_model", 256), config.get("d_model", 256))
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.get("d_model", 256))
        self.decoder = nn.Linear(config.get("d_model", 256), config.get("vocab_size", 10000))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLM prediction.

        Args:
            hidden_states: Output from the encoder [batch_size, seq_len, d_model]

        Returns:
            Prediction scores (logits) for each token [batch_size, seq_len, vocab_size]
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class EncoderOnlyModel(BaseModel):
    """
    An encoder-only transformer model, similar to BERT, for tasks like MLM.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the encoder-only model.

        Args:
            config: Model configuration parameters
        """
        super().__init__(config)

        # Extract configuration parameters
        self.vocab_size = config.get("vocab_size", 10000)
        self.d_model = config.get("d_model", 256)
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 4)
        self.d_ff = config.get("d_ff", self.d_model * 4)
        self.max_seq_len = config.get("max_seq_len", 512)
        self.dropout = config.get("dropout", 0.1)
        # BERT often uses type embeddings (segment embeddings), add if needed
        # self.type_vocab_size = config.get("type_vocab_size", 2)

        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        )

        # Optional: Segment embedding (for tasks like Next Sentence Prediction)
        # self.segment_embedding = nn.Embedding(self.type_vocab_size, self.d_model)

        # Layer normalization and dropout for embeddings
        self.embedding_layer_norm = nn.LayerNorm(self.d_model)
        self.embedding_dropout = nn.Dropout(self.dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])

        # MLM prediction head
        self.mlm_head = BertMLMHead(config)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Special initialization for embeddings if needed
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_encoding.pe, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the encoder model.

        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens [batch_size, seq_len]
            token_type_ids: Optional segment IDs [batch_size, seq_len]

        Returns:
            Sequence output: Final hidden states from the encoder [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Convert attention mask to format expected by transformer [batch_size, 1, seq_len]
        # For multi-head attention, it expects [batch_size, num_heads, seq_len, seq_len] or [batch_size, 1, 1, seq_len] for broadcasting
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # Use large negative value for masking


        # Embed tokens and add positional encoding
        token_embeddings = self.token_embedding(input_ids)
        positional_embeddings = self.positional_encoding.pe[:, :seq_len]
        embeddings = token_embeddings + positional_embeddings

        # Add segment embeddings if provided
        # if token_type_ids is not None:
        #     segment_embeddings = self.segment_embedding(token_type_ids)
        #     embeddings += segment_embeddings

        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        # Apply transformer encoder layers
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            # The mask format expected by MultiHeadAttention is [batch_size, 1, seq_len]
            # Let's adjust the mask format here if needed, or ensure the component handles it.
            # The current TransformerEncoderLayer expects [batch_size, 1, seq_len]
            layer_attention_mask = attention_mask.unsqueeze(1) # Shape: [batch_size, 1, seq_len]
            hidden_states = encoder_layer(hidden_states, mask=layer_attention_mask)

        return hidden_states

    def predict_mlm(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the encoder and MLM head to get predictions.

        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens [batch_size, seq_len]
            token_type_ids: Optional segment IDs [batch_size, seq_len]

        Returns:
            MLM prediction logits [batch_size, seq_len, vocab_size]
        """
        # Get hidden states from the encoder
        hidden_states = self.forward(input_ids, attention_mask, token_type_ids)

        # Pass hidden states through the MLM head
        mlm_logits = self.mlm_head(hidden_states)

        return mlm_logits

    # Note: A standard encoder-only model like BERT doesn't typically have a `generate` method.
    # Generation is usually handled by decoder or encoder-decoder models.
    # If generation is needed based on the encoder output, it would require a different setup.

