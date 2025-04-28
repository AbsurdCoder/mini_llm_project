"""
Model extraction and retraining utilities.

This module provides functions for extracting trained models and continuing training from checkpoints.
"""

import os
import json
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Union

from models.transformer_model import TransformerModel, DecoderOnlyTransformer
from models.encoder_model import EncoderOnlyModel
from tokenizers.bpe_tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer
from training.enhanced_trainer import Trainer

# Set up logger
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_path: str, 
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[Union[TransformerModel, DecoderOnlyTransformer, EncoderOnlyModel], Union[BPETokenizer, CharacterTokenizer]]:
    """
    Load a model and tokenizer from checkpoint files.
    
    Args:
        model_path: Path to the model checkpoint (without extension)
        tokenizer_path: Path to the tokenizer file (if None, will try to infer from model path)
        device: Device to load the model to ('cpu', 'cuda', 'mps', or None for auto-detection)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set up device
    if device is None:
        # Try to use MPS (Mac GPU) if available
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS with a small operation
                test_tensor = torch.ones(1, device='mps')
                _ = test_tensor * 2
                device = 'mps'
            except:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading model from {model_path} to {device}")
    
    # Load model config
    config_path = f"{model_path}_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine model type from config
    model_type = config.get("model_type", "decoder_only")
    
    # Create model instance based on type
    if model_type == "encoder_only":
        model = EncoderOnlyModel(config)
    elif model_type == "transformer":
        model = TransformerModel(config)
    else:  # Default to decoder_only
        model = DecoderOnlyTransformer(config)
    
    # Load model weights
    checkpoint_path = f"{model_path}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Load tokenizer
    if tokenizer_path is None:
        # Try to infer tokenizer path from model path
        model_dir = os.path.dirname(model_path)
        tokenizer_path = os.path.join(model_dir, "..", "data", "tokenizer.json")
        
        if not os.path.exists(tokenizer_path):
            # Try alternative locations
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
            if not os.path.exists(tokenizer_path):
                tokenizer_path = os.path.join("./data", "tokenizer.json")
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"Could not find tokenizer file. Please specify tokenizer_path.")
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    # Determine tokenizer type from config or try both
    tokenizer_type = config.get("tokenizer_type", "bpe")
    
    try:
        if tokenizer_type.lower() == "bpe":
            tokenizer = BPETokenizer.load(tokenizer_path)
        elif tokenizer_type.lower() == "character":
            tokenizer = CharacterTokenizer.load(tokenizer_path)
        else:
            # Try BPE first, then character
            try:
                tokenizer = BPETokenizer.load(tokenizer_path)
            except:
                tokenizer = CharacterTokenizer.load(tokenizer_path)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer: {str(e)}")
    
    return model, tokenizer

def continue_training(
    model_path: str,
    train_dataloader,
    val_dataloader=None,
    learning_rate: float = 1e-4,  # Lower learning rate for fine-tuning
    epochs: int = 5,
    checkpoint_dir: str = "./checkpoints/continued",
    device: Optional[str] = None,
    early_stopping_patience: Optional[int] = None
) -> Dict[str, Any]:
    """
    Continue training a model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (without extension)
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        learning_rate: Learning rate for optimizer (typically lower for continued training)
        epochs: Number of epochs to train for
        checkpoint_dir: Directory to save new checkpoints
        device: Device to train on ('cpu', 'cuda', 'mps', or None for auto-detection)
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        
    Returns:
        Dictionary with training results
    """
    # Load model
    logger.info(f"Loading model from {model_path} for continued training")
    
    # Load model config
    config_path = f"{model_path}_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine model type from config
    model_type = config.get("model_type", "decoder_only")
    
    # Create model instance based on type
    if model_type == "encoder_only":
        model = EncoderOnlyModel(config)
    elif model_type == "transformer":
        model = TransformerModel(config)
    else:  # Default to decoder_only
        model = DecoderOnlyTransformer(config)
    
    # Load model weights
    checkpoint_path = f"{model_path}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU first
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        device=device,
        early_stopping_patience=early_stopping_patience
    )
    
    # Continue training
    logger.info(f"Continuing training for {epochs} epochs")
    results = trainer.train(epochs)
    
    # Save config file with the correct name for the UI
    final_model_path = os.path.join(checkpoint_dir, "best_model")
    config_path = f"{final_model_path}_config.json"
    with open(config_path, "w") as f:
        json.dump(model.config, f, indent=2)
    
    return results

def extract_model_for_inference(
    model_path: str,
    output_path: str,
    device: Optional[str] = None
) -> str:
    """
    Extract a model from a checkpoint and save it in a format optimized for inference.
    
    Args:
        model_path: Path to the model checkpoint (without extension)
        output_path: Path to save the extracted model (without extension)
        device: Device to load the model to ('cpu', 'cuda', 'mps', or None for auto-detection)
        
    Returns:
        Path to the extracted model
    """
    # Load model
    logger.info(f"Extracting model from {model_path} for inference")
    
    # Load model config
    config_path = f"{model_path}_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine model type from config
    model_type = config.get("model_type", "decoder_only")
    
    # Create model instance based on type
    if model_type == "encoder_only":
        model = EncoderOnlyModel(config)
    elif model_type == "transformer":
        model = TransformerModel(config)
    else:  # Default to decoder_only
        model = DecoderOnlyTransformer(config)
    
    # Load model weights
    checkpoint_path = f"{model_path}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU first
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model in inference format (just the state dict)
    torch.save(model.state_dict(), f"{output_path}.pt")
    
    # Save config
    with open(f"{output_path}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model extracted and saved to {output_path}.pt")
    
    return output_path
