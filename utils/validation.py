"""
Validation utilities for the Mini LLM project.

This module provides functions for validating inputs, configurations, and parameters
to ensure proper operation of the Mini LLM project components.
"""

import os
import re
import logging
import torch
from typing import Dict, Any, List, Optional, Union, Tuple

# Set up logger
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

def validate_file_path(file_path: str, must_exist: bool = True, 
                      allowed_extensions: Optional[List[str]] = None) -> str:
    """
    Validate a file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.pdf'])
        
    Returns:
        Validated file path
        
    Raises:
        ValidationError: If validation fails
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    # Check if path exists
    if must_exist and not os.path.exists(file_path):
        raise ValidationError(f"File not found: {file_path}")
    
    # Check file extension
    if allowed_extensions:
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension: {ext}. Allowed extensions: {', '.join(allowed_extensions)}"
            )
    
    return file_path

def validate_directory(directory: str, create_if_missing: bool = True) -> str:
    """
    Validate a directory path.
    
    Args:
        directory: Directory path to validate
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        Validated directory path
        
    Raises:
        ValidationError: If validation fails
    """
    if not directory:
        raise ValidationError("Directory path cannot be empty")
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        if create_if_missing:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                raise ValidationError(f"Failed to create directory {directory}: {str(e)}")
        else:
            raise ValidationError(f"Directory not found: {directory}")
    
    # Check if it's a directory
    if not os.path.isdir(directory):
        raise ValidationError(f"Not a directory: {directory}")
    
    return directory

def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If validation fails
    """
    required_keys = ["vocab_size", "d_model", "num_heads", "num_layers", "d_ff", "max_seq_len", "dropout"]
    
    # Check for required keys
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"Missing required configuration key: {key}")
    
    # Validate values
    if config["vocab_size"] <= 0:
        raise ValidationError(f"Invalid vocab_size: {config['vocab_size']}. Must be positive.")
    
    if config["d_model"] <= 0:
        raise ValidationError(f"Invalid d_model: {config['d_model']}. Must be positive.")
    
    if config["num_heads"] <= 0:
        raise ValidationError(f"Invalid num_heads: {config['num_heads']}. Must be positive.")
    
    if config["d_model"] % config["num_heads"] != 0:
        raise ValidationError(
            f"d_model ({config['d_model']}) must be divisible by num_heads ({config['num_heads']})"
        )
    
    if config["num_layers"] <= 0:
        raise ValidationError(f"Invalid num_layers: {config['num_layers']}. Must be positive.")
    
    if config["d_ff"] <= 0:
        raise ValidationError(f"Invalid d_ff: {config['d_ff']}. Must be positive.")
    
    if config["max_seq_len"] <= 0:
        raise ValidationError(f"Invalid max_seq_len: {config['max_seq_len']}. Must be positive.")
    
    if not 0 <= config["dropout"] < 1:
        raise ValidationError(f"Invalid dropout: {config['dropout']}. Must be between 0 and 1.")
    
    return config

def validate_training_params(
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    optimizer: str,
    scheduler: str,
    early_stopping_patience: Optional[int]
) -> Dict[str, Any]:
    """
    Validate training parameters.
    
    Args:
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        optimizer: Optimizer type
        scheduler: Scheduler type
        early_stopping_patience: Early stopping patience
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    if batch_size <= 0:
        raise ValidationError(f"Invalid batch_size: {batch_size}. Must be positive.")
    
    if num_epochs <= 0:
        raise ValidationError(f"Invalid num_epochs: {num_epochs}. Must be positive.")
    
    if learning_rate <= 0:
        raise ValidationError(f"Invalid learning_rate: {learning_rate}. Must be positive.")
    
    valid_optimizers = ["adam", "adamw", "sgd"]
    if optimizer not in valid_optimizers:
        raise ValidationError(f"Invalid optimizer: {optimizer}. Must be one of {valid_optimizers}")
    
    valid_schedulers = ["linear", "cosine", "constant", "step", "none"]
    if scheduler not in valid_schedulers:
        raise ValidationError(f"Invalid scheduler: {scheduler}. Must be one of {valid_schedulers}")
    
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValidationError(
            f"Invalid early_stopping_patience: {early_stopping_patience}. Must be positive."
        )
    
    return {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "early_stopping_patience": early_stopping_patience
    }

def validate_generation_params(
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float
) -> Dict[str, Any]:
    """
    Validate text generation parameters.
    
    Args:
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    if max_length <= 0:
        raise ValidationError(f"Invalid max_length: {max_length}. Must be positive.")
    
    if temperature <= 0:
        raise ValidationError(f"Invalid temperature: {temperature}. Must be positive.")
    
    if top_k <= 0:
        raise ValidationError(f"Invalid top_k: {top_k}. Must be positive.")
    
    if not 0 < top_p <= 1:
        raise ValidationError(f"Invalid top_p: {top_p}. Must be between 0 and 1.")
    
    return {
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    }

def validate_device(device: str) -> str:
    """
    Validate and normalize device specification.
    
    Args:
        device: Device specification ('cpu', 'cuda', 'mps', or '')
        
    Returns:
        Validated device string
        
    Raises:
        ValidationError: If validation fails
    """
    if not device:
        # Auto-detect device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Test MPS with a small operation
                test_tensor = torch.ones(1, device='mps')
                _ = test_tensor * 2
                return "mps"
            except:
                return "cpu"
        else:
            return "cpu"
    
    # Validate specified device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    
    if device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
        try:
            # Test MPS with a small operation
            test_tensor = torch.ones(1, device='mps')
            _ = test_tensor * 2
            return "mps"
        except:
            logger.warning("MPS requested but failed to initialize. Falling back to CPU.")
            return "cpu"
    
    if device not in ["cpu", "cuda", "mps"]:
        raise ValidationError(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'.")
    
    return device

def validate_data_split_params(
    val_ratio: float,
    test_ratio: float,
    split_method: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    split_regex: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate data splitting parameters.
    
    Args:
        val_ratio: Validation data ratio
        test_ratio: Test data ratio
        split_method: Text splitting method
        chunk_size: Size of chunks when using 'chunks' split method
        chunk_overlap: Overlap between chunks when using 'chunks' split method
        split_regex: Regex pattern for splitting when using 'regex' split method
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    if not 0 <= val_ratio < 1:
        raise ValidationError(f"Invalid val_ratio: {val_ratio}. Must be between 0 and 1.")
    
    if not 0 <= test_ratio < 1:
        raise ValidationError(f"Invalid test_ratio: {test_ratio}. Must be between 0 and 1.")
    
    if val_ratio + test_ratio >= 1:
        raise ValidationError(
            f"Sum of val_ratio ({val_ratio}) and test_ratio ({test_ratio}) must be less than 1."
        )
    
    valid_split_methods = ["paragraphs", "sentences", "chunks", "headings", "regex"]
    if split_method not in valid_split_methods:
        raise ValidationError(
            f"Invalid split_method: {split_method}. Must be one of {valid_split_methods}"
        )
    
    if split_method == "chunks":
        if chunk_size is None or chunk_size <= 0:
            raise ValidationError(f"Invalid chunk_size: {chunk_size}. Must be positive.")
        
        if chunk_overlap is None or chunk_overlap < 0:
            raise ValidationError(f"Invalid chunk_overlap: {chunk_overlap}. Must be non-negative.")
        
        if chunk_overlap >= chunk_size:
            raise ValidationError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
            )
    
    if split_method == "regex" and not split_regex:
        raise ValidationError("split_regex must be provided when using 'regex' split method.")
    
    return {
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_method": split_method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "split_regex": split_regex
    }

def validate_tokenizer_params(
    tokenizer_type: str,
    vocab_size: int,
    tokenizer_path: str
) -> Dict[str, Any]:
    """
    Validate tokenizer parameters.
    
    Args:
        tokenizer_type: Type of tokenizer ('bpe' or 'character')
        vocab_size: Vocabulary size
        tokenizer_path: Path to save/load tokenizer
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    valid_tokenizer_types = ["bpe", "character"]
    if tokenizer_type not in valid_tokenizer_types:
        raise ValidationError(
            f"Invalid tokenizer_type: {tokenizer_type}. Must be one of {valid_tokenizer_types}"
        )
    
    if vocab_size <= 0:
        raise ValidationError(f"Invalid vocab_size: {vocab_size}. Must be positive.")
    
    if not tokenizer_path:
        raise ValidationError("tokenizer_path cannot be empty.")
    
    return {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "tokenizer_path": tokenizer_path
    }

def validate_text_data(texts: List[str], min_samples: int = 10) -> List[str]:
    """
    Validate text data for training.
    
    Args:
        texts: List of text samples
        min_samples: Minimum number of samples required
        
    Returns:
        Validated text data
        
    Raises:
        ValidationError: If validation fails
    """
    if not texts:
        raise ValidationError("Text data cannot be empty.")
    
    if len(texts) < min_samples:
        raise ValidationError(
            f"Insufficient text samples: {len(texts)}. At least {min_samples} samples required."
        )
    
    # Filter out empty samples
    valid_texts = [text for text in texts if text.strip()]
    
    if len(valid_texts) < min_samples:
        raise ValidationError(
            f"Insufficient non-empty text samples: {len(valid_texts)}. "
            f"At least {min_samples} samples required."
        )
    
    return valid_texts

def validate_args(args) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Command line arguments
        
    Raises:
        ValidationError: If validation fails
    """
    # Validate mode
    valid_modes = ["train", "retrain", "test", "generate"]
    if args.mode not in valid_modes:
        raise ValidationError(f"Invalid mode: {args.mode}. Must be one of {valid_modes}")
    
    # Validate device
    validate_device(args.device)
    
    # Validate data path for modes that require it
    if args.mode in ["train", "retrain", "test"]:
        validate_file_path(
            args.data_path,
            must_exist=True,
            allowed_extensions=[".txt", ".html", ".pdf", ".json"]
        )
    
    # Validate model path for modes that require it
    if args.mode in ["retrain", "test", "generate"]:
        model_dir = os.path.dirname(args.model_path)
        validate_directory(model_dir, create_if_missing=False)
        
        # Check for model files
        model_pt = f"{args.model_path}.pt"
        model_config = f"{args.model_path}_config.json"
        
        if not os.path.exists(model_pt):
            raise ValidationError(f"Model checkpoint not found: {model_pt}")
        
        if not os.path.exists(model_config):
            raise ValidationError(f"Model config not found: {model_config}")
    
    # Validate checkpoint directory
    validate_directory(args.checkpoint_dir, create_if_missing=True)
    
    # Validate data split parameters
    validate_data_split_params(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_method=args.split_method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        split_regex=args.split_regex
    )
    
    # Validate tokenizer parameters
    validate_tokenizer_params(
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        tokenizer_path=args.tokenizer_path
    )
    
    # Validate training parameters
    if args.mode in ["train", "retrain"]:
        validate_training_params(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            early_stopping_patience=args.early_stopping_patience
        )
    
    # Validate generation parameters
    if args.mode == "generate":
        validate_generation_params(
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    # Validate test output directory
    if args.mode == "test":
        validate_directory(args.test_output_dir, create_if_missing=True)
