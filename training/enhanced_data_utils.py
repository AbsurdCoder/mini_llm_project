"""
Enhanced dataset and DataLoader utilities for training language models.
Supports multiple file formats (TXT, PDF, HTML) and flexible text splitting options.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
import random
import os
import re
from pathlib import Path

# Import libraries for handling different file formats
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import html2text


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


def load_file(file_path: str) -> str:
    """
    Load content from a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Text content of the file
    
    Raises:
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if extension == '.txt':
        # Load plain text file
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif extension == '.pdf':
        # Load PDF file
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        return text
    
    elif extension in ['.html', '.htm']:
        # Load HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Convert to plain text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_tables = False
        text = h.handle(str(soup))
        
        return text
    
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def split_text(text: str, split_method: str = 'paragraph', 
               min_length: int = 50, max_length: int = 10000) -> List[str]:
    """
    Split text into samples using different methods.
    
    Args:
        text: Input text to split
        split_method: Method to use for splitting:
            - 'paragraph': Split by double newlines
            - 'sentence': Split by sentences
            - 'chunk': Split by fixed chunk size
            - 'semantic': Split by semantic boundaries (headings, sections)
            - 'regex': Split by custom regex pattern
        min_length: Minimum length of a sample (characters)
        max_length: Maximum length of a sample (characters)
        
    Returns:
        List of text samples
    """
    if not text:
        return []
    
    samples = []
    
    if split_method == 'paragraph':
        # Split by paragraphs (double newlines)
        raw_samples = re.split(r'\n\s*\n', text)
    
    elif split_method == 'sentence':
        # Split by sentences (using regex for sentence boundaries)
        # This is a simplified sentence splitter
        raw_samples = re.split(r'(?<=[.!?])\s+', text)
    
    elif split_method == 'chunk':
        # Split by fixed chunk size (with overlap)
        chunk_size = max_length
        overlap = chunk_size // 4  # 25% overlap
        
        raw_samples = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= min_length:
                raw_samples.append(chunk)
        
        return raw_samples  # Already filtered by length
    
    elif split_method == 'semantic':
        # Split by semantic boundaries (headings, sections)
        # Look for headings, section markers, etc.
        heading_pattern = r'(?:\n|^)(?:#+ .*|\d+\.\s+.*|[A-Z][A-Z\s]+:|\*\*.*\*\*|\=+\s+.*\s+\=+)'
        raw_samples = re.split(heading_pattern, text)
        
        # Recombine heading with its content
        combined_samples = []
        for i in range(1, len(raw_samples)):
            # Find the heading that was split
            heading_match = re.search(heading_pattern, text)
            heading = heading_match.group(0) if heading_match else ""
            
            # Combine heading with content
            if heading and raw_samples[i]:
                combined_samples.append(heading + raw_samples[i])
            elif raw_samples[i]:
                combined_samples.append(raw_samples[i])
        
        raw_samples = combined_samples if combined_samples else raw_samples
    
    elif split_method == 'regex':
        # Split by custom regex pattern (default to paragraph)
        # Users can override this with their own pattern
        pattern = r'\n\s*\n'
        raw_samples = re.split(pattern, text)
    
    else:
        # Default to paragraph splitting
        raw_samples = re.split(r'\n\s*\n', text)
    
    # Filter samples by length
    for sample in raw_samples:
        sample = sample.strip()
        if len(sample) >= min_length and len(sample) <= max_length:
            samples.append(sample)
    
    return samples


def load_and_split_data(
    file_path: str,
    split_method: str = 'paragraph',
    min_length: int = 50,
    max_length: int = 10000,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load data from a file and split into train, validation, and test sets.
    
    Args:
        file_path: Path to the data file
        split_method: Method to use for splitting text into samples
        min_length: Minimum length of a sample (characters)
        max_length: Maximum length of a sample (characters)
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load file content
    text = load_file(file_path)
    
    # Split text into samples
    samples = split_text(
        text=text,
        split_method=split_method,
        min_length=min_length,
        max_length=max_length
    )
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Split into train, validation, and test sets
    n_samples = len(samples)
    n_val = max(1, int(n_samples * val_split))
    n_test = max(1, int(n_samples * test_split))
    n_train = n_samples - n_val - n_test
    
    train_texts = samples[:n_train]
    val_texts = samples[n_train:n_train + n_val]
    test_texts = samples[n_train + n_val:]
    
    return train_texts, val_texts, test_texts


def load_multiple_files(
    file_paths: List[str],
    split_method: str = 'paragraph',
    min_length: int = 50,
    max_length: int = 10000,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load data from multiple files and split into train, validation, and test sets.
    
    Args:
        file_paths: List of paths to data files
        split_method: Method to use for splitting text into samples
        min_length: Minimum length of a sample (characters)
        max_length: Maximum length of a sample (characters)
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    all_samples = []
    
    # Load and process each file
    for file_path in file_paths:
        # Load file content
        text = load_file(file_path)
        
        # Split text into samples
        samples = split_text(
            text=text,
            split_method=split_method,
            min_length=min_length,
            max_length=max_length
        )
        
        all_samples.extend(samples)
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    # Split into train, validation, and test sets
    n_samples = len(all_samples)
    n_val = max(1, int(n_samples * val_split))
    n_test = max(1, int(n_samples * test_split))
    n_train = n_samples - n_val - n_test
    
    train_texts = all_samples[:n_train]
    val_texts = all_samples[n_train:n_train + n_val]
    test_texts = all_samples[n_train + n_val:]
    
    return train_texts, val_texts, test_texts
