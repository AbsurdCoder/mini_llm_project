"""
Base model class that defines the interface for all transformer models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for transformer models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Model configuration parameters
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens [batch_size, seq_len]
            
        Returns:
            Output tensor
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model weights and configuration to a file.
        
        Args:
            path: Path to save the model
        """
        import os
        import json
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), f"{path}.pt")
        
        # Save model config
        with open(f"{path}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'BaseModel':
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to ('cpu' or 'cuda')
            
        Returns:
            Loaded model instance
        """
        import json
        
        # Load model config
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(config)
        
        # Load model weights
        model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
        model.to(device)
        
        return model
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params
        }
