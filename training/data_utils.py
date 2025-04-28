"""
Dataset and DataLoader utilities for training language models.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import random


class TextDataset(Dataset):
    """Dataset for language modeling tasks."""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer, 
                 max_length: int = 512,
                 is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer instance for encoding texts
            max_length: Maximum sequence length
            is_training: Whether this dataset is for training
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.texts[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length - 2:  # -2 for BOS and EOS tokens
            token_ids = token_ids[:self.max_length - 2]
        
        # Add special tokens
        token_ids = [self.tokenizer.token_to_id["[BOS]"]] + token_ids + [self.tokenizer.token_to_id["[EOS]"]]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.token_to_id["[PAD]"]] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # For language modeling, labels are the same as input_ids (shifted by 1)
        labels = input_ids.clone()
        if self.is_training:
            labels[:-1] = input_ids[1:]  # Shift left by 1
            labels[-1] = self.tokenizer.token_to_id["[PAD]"]  # Last token predicts padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_dataloaders(
    train_texts: List[str],
    val_texts: List[str],
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_texts: List of training text samples
        val_texts: List of validation text samples
        tokenizer: Tokenizer instance for encoding texts
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=True
    )
    
    val_dataset = TextDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=True
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader
