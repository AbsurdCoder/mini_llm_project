"""
Utility functions for the mini LLM project.
"""
import os
import json
import torch
import numpy as np
import random
from typing import Optional, Dict, Any, List


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    Get the device to use for training/inference.
    
    Returns:
        torch.device: Device to use
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save the data to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        path: Path to load the data from
        
    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_text_file(path: str) -> str:
    """
    Load text from a file.
    
    Args:
        path: Path to the text file
        
    Returns:
        Text content
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_text_file(text: str, path: str) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text to save
        path: Path to save the text to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def split_dataset(texts: List[str], val_ratio: float = 0.1, test_ratio: float = 0.1, 
                 shuffle: bool = True, seed: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Split a dataset into training, validation, and test sets.
    
    Args:
        texts: List of text samples
        val_ratio: Ratio of validation samples
        test_ratio: Ratio of test samples
        shuffle: Whether to shuffle the data before splitting
        seed: Random seed for shuffling
        
    Returns:
        Dictionary with 'train', 'val', and 'test' splits
    """
    if seed is not None:
        random.seed(seed)
    
    if shuffle:
        texts_copy = texts.copy()
        random.shuffle(texts_copy)
    else:
        texts_copy = texts
    
    n_samples = len(texts_copy)
    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_val - n_test
    
    train_texts = texts_copy[:n_train]
    val_texts = texts_copy[n_train:n_train + n_val]
    test_texts = texts_copy[n_train + n_val:]
    
    return {
        'train': train_texts,
        'val': val_texts,
        'test': test_texts
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }
