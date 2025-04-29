"""
Unified main script for the Mini LLM project.
Provides a comprehensive CLI interface for all model operations.

Features:
- Multi-format data loading (.txt, .html, .pdf, .json)
- Multiple model architectures (Transformer, Encoder-only, Decoder-only)
- Detailed logging of model internals and training progress
- Model training, retraining, testing, and generation
- Comprehensive error handling and validation

Usage:
    python main.py --mode train --data_path ./data/corpus.txt
    python main.py --mode retrain --model_path ./checkpoints/best_model --data_path ./data/new_corpus.txt
    python main.py --mode test --model_path ./checkpoints/best_model --data_path ./data/test_corpus.txt
    python main.py --mode generate --model_path ./checkpoints/best_model --prompt "Once upon a time"
"""
import os
import sys
import argparse
import torch
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm

# Import tokenizers
from tokenizers.bpe_tokenizer import BPETokenizer
from tokenizers.character_tokenizer import CharacterTokenizer

# Import models
from models.transformer_model import TransformerModel, DecoderOnlyTransformer
from models.encoder_model import EncoderOnlyModel
from models.enhanced_transformer_components import (
    MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding,
    TransformerEncoderLayer, TransformerDecoderLayer
)

# Import training utilities
from training.enhanced_detailed_trainer import EnhancedTrainer
from training.model_extraction import load_model_and_tokenizer, continue_training, extract_model_for_inference

# Import testing utilities
from testing.model_tester import ModelTester

# Import utilities
from utils.helpers import set_seed
from utils.file_loader import FileLoader, DataSplitter


def setup_logging(verbose=False, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable debug logging
        log_file: Optional path to log file
        
    Returns:
        Logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Mini LLM Project - Unified CLI Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train", "retrain", "test", "generate"],
                        help="Operation mode: train, retrain, test, or generate")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data/corpus.txt",
                        help="Path to the data file (supports .txt, .html, .pdf, .json)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of test data")
    parser.add_argument("--split_method", type=str, default="paragraphs",
                        choices=["paragraphs", "sentences", "chunks", "headings", "regex"],
                        help="Method to split text into samples")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Size of chunks when using 'chunks' split method")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Overlap between chunks when using 'chunks' split method")
    parser.add_argument("--split_regex", type=str, default=r"\n\n+",
                        help="Regex pattern for splitting when using 'regex' split method")
    
    # Tokenizer arguments
    parser.add_argument("--tokenizer_type", type=str, default="bpe", 
                        choices=["bpe", "character"],
                        help="Type of tokenizer to use")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size for the tokenizer")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizer.json",
                        help="Path to save/load the tokenizer")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="decoder_only", 
                        choices=["transformer", "decoder_only", "encoder_only"],
                        help="Type of model to use")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer to use")
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "constant", "step", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of epochs with no improvement after which training will be stopped")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    
    # Testing arguments
    parser.add_argument("--test_output_dir", type=str, default="./test_results",
                        help="Directory to save test results")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="",
                        choices=["cpu", "cuda", "mps", ""],
                        help="Device to use (empty for auto-detection)")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model",
                        help="Path to save/load the model")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging (including model internals)")
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to log file (if empty, logs to console only)")
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """
    Determine the device to use.
    
    Args:
        device_arg: Device argument from command line
        
    Returns:
        Device string ('cpu', 'cuda', or 'mps')
    """
    if device_arg:
        return device_arg
        
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


def load_and_split_data(
    data_path: str,
    split_method: str,
    val_ratio: float,
    test_ratio: float,
    chunk_size: int,
    chunk_overlap: int,
    split_regex: str,
    seed: int,
    logger: logging.Logger
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load and split data into train, validation, and test sets.
    
    Args:
        data_path: Path to the data file
        split_method: Method to split text into samples
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        chunk_size: Size of chunks when using 'chunks' split method
        chunk_overlap: Overlap between chunks when using 'chunks' split method
        split_regex: Regex pattern for splitting when using 'regex' split method
        seed: Random seed
        logger: Logger instance
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        # Use the enhanced file loader to support multiple formats
        text_data = FileLoader.load_file(data_path)
        logger.info(f"Loaded {len(text_data):,} characters of text data")
        
        # Split text into samples using the specified method
        logger.info(f"Splitting text using method: {split_method}")
        if split_method == "paragraphs":
            samples = DataSplitter.split_by_paragraphs(text_data)
        elif split_method == "sentences":
            samples = DataSplitter.split_by_sentences(text_data)
        elif split_method == "chunks":
            samples = DataSplitter.split_by_chunks(
                text_data, 
                chunk_size=chunk_size, 
                overlap=chunk_overlap
            )
        elif split_method == "headings":
            samples = DataSplitter.split_by_headings(text_data)
        elif split_method == "regex":
            samples = DataSplitter.split_by_regex(text_data, split_regex)
        
        logger.info(f"Created {len(samples):,} text samples")
        
        # Split dataset into train/val/test
        train_size = int(len(samples) * (1 - val_ratio - test_ratio))
        val_size = int(len(samples) * val_ratio)
        
        # Shuffle samples
        import random
        random.seed(seed)
        random.shuffle(samples)
        
        train_texts = samples[:train_size]
        val_texts = samples[train_size:train_size + val_size]
        test_texts = samples[train_size + val_size:]
        
        logger.info(f"Split dataset: {len(train_texts):,} train, {len(val_texts):,} val, {len(test_texts):,} test")
        
        # Save splits info for reference
        splits_info = {
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
            "split_method": split_method,
            "data_path": data_path
        }
        
        os.makedirs("./data", exist_ok=True)
        with open("./data/data_splits.json", "w") as f:
            json.dump(splits_info, f, indent=2)
        
        return train_texts, val_texts, test_texts
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def create_or_load_tokenizer(
    tokenizer_type: str,
    tokenizer_path: str,
    vocab_size: int,
    train_texts: List[str],
    mode: str,
    logger: logging.Logger
) -> Union[BPETokenizer, CharacterTokenizer]:
    """
    Create a new tokenizer or load an existing one.
    
    Args:
        tokenizer_type: Type of tokenizer ('bpe' or 'character')
        tokenizer_path: Path to save/load the tokenizer
        vocab_size: Vocabulary size for the tokenizer
        train_texts: Training texts for tokenizer training
        mode: Operation mode ('train', 'retrain', 'test', or 'generate')
        logger: Logger instance
        
    Returns:
        Tokenizer instance
    """
    # For training mode, create and train a new tokenizer
    if mode == "train":
        logger.info(f"Creating {tokenizer_type} tokenizer with vocab size {vocab_size}")
        
        try:
            if tokenizer_type == "bpe":
                tokenizer = BPETokenizer(vocab_size=vocab_size)
            else:
                tokenizer = CharacterTokenizer(vocab_size=vocab_size)
            
            logger.info("Training tokenizer on data")
            tokenizer.train(train_texts)
            
            # Save tokenizer
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
            tokenizer.save(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error creating tokenizer: {str(e)}")
            raise
    
    # For other modes, load an existing tokenizer
    else:
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        
        try:
            if tokenizer_type == "bpe":
                tokenizer = BPETokenizer.load(tokenizer_path)
            else:
                tokenizer = CharacterTokenizer.load(tokenizer_path)
                
            logger.info(f"Tokenizer loaded with vocabulary size {len(tokenizer.token_to_id)}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise


def create_model(
    model_type: str,
    vocab_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    max_seq_len: int,
    dropout: float,
    verbose: bool,
    checkpoint_dir: str,
    logger: logging.Logger
) -> Union[TransformerModel, DecoderOnlyTransformer, EncoderOnlyModel]:
    """
    Create a new model.
    
    Args:
        model_type: Type of model ('transformer', 'decoder_only', or 'encoder_only')
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        verbose: Whether to enable verbose logging
        checkpoint_dir: Directory to save model config
        logger: Logger instance
        
    Returns:
        Model instance
    """
    logger.info(f"Creating {model_type} model")
    
    try:
        # Prepare model config
        model_config = {
            "model_type": model_type,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "verbose": verbose  # Pass verbose flag to model for detailed logging
        }
        
        if model_type == "transformer":
            model = TransformerModel(model_config)
        elif model_type == "decoder_only":
            model = DecoderOnlyTransformer(model_config)
        elif model_type == "encoder_only":
            model = EncoderOnlyModel(model_config)
        
        # Get parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
        
        # Save model configuration
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise


def prepare_data_for_training(
    train_texts: List[str],
    val_texts: List[str],
    tokenizer: Union[BPETokenizer, CharacterTokenizer],
    batch_size: int,
    max_length: int,
    device: str,
    logger: logging.Logger
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepare data for training.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        device: Device to use
        logger: Logger instance
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info("Preparing data for training")
    
    try:
        from torch.utils.data import Dataset, DataLoader
        
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # Tokenize text
                tokens = self.tokenizer.encode(text)
                
                # Truncate or pad to max_length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                # Create input_ids and labels
                input_ids = tokens[:-1] if len(tokens) > 1 else tokens
                labels = tokens[1:] if len(tokens) > 1 else tokens
                
                # Pad to max_length - 1 (since we're using input_ids and labels)
                pad_length = self.max_length - 1 - len(input_ids)
                if pad_length > 0:
                    input_ids = input_ids + [0] * pad_length
                    labels = labels + [0] * pad_length
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * (len(tokens) - pad_length - 1) + [0] * pad_length
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
                }
        
        # Create datasets
        train_dataset = TextDataset(train_texts, tokenizer, max_length)
        val_dataset = TextDataset(val_texts, tokenizer, max_length) if val_texts else None
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        logger.info(f"Created training dataloader with {len(train_dataloader)} batches")
        if val_dataloader:
            logger.info(f"Created validation dataloader with {len(val_dataloader)} batches")
        
        return train_dataloader, val_dataloader
        
    except Exception as e:
        logger.error(f"Error preparing data for training: {str(e)}")
        raise


def train_model_with_enhanced_trainer(
    model: Union[TransformerModel, DecoderOnlyTransformer, EncoderOnlyModel],
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    learning_rate: float,
    num_epochs: int,
    checkpoint_dir: str,
    device: str,
    early_stopping_patience: Optional[int],
    verbose: bool,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Train model using the enhanced trainer.
    
    Args:
        model: Model instance
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        learning_rate: Learning rate
        num_epochs: Number of epochs
        checkpoint_dir: Directory to save checkpoints
        device: Device to use
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        verbose: Whether to enable verbose logging
        logger: Logger instance
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting model training with enhanced trainer")
    
    try:
        # Create trainer
        trainer = EnhancedTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            device=device,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Train model
        results = trainer.train(num_epochs)
        
        logger.info(f"Training completed with {results['epochs_completed']} epochs")
        if val_dataloader:
            logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        else:
            logger.info(f"Best training loss: {results['best_train_loss']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def train(args, logger):
    """
    Train a new model.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("STARTING TRAINING MODE")
    logger.info("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load and split data
        train_texts, val_texts, test_texts = load_and_split_data(
            data_path=args.data_path,
            split_method=args.split_method,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split_regex=args.split_regex,
            seed=args.seed,
            logger=logger
        )
        
        # Create and train tokenizer
        tokenizer = create_or_load_tokenizer(
            tokenizer_type=args.tokenizer_type,
            tokenizer_path=args.tokenizer_path,
            vocab_size=args.vocab_size,
            train_texts=train_texts,
            mode="train",
            logger=logger
        )
        
        # Create model
        model = create_model(
            model_type=args.model_type,
            vocab_size=len(tokenizer.token_to_id),
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            verbose=args.verbose,
            checkpoint_dir=args.checkpoint_dir,
            logger=logger
        )
        
        # Prepare data for training
        train_dataloader, val_dataloader = prepare_data_for_training(
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
            device=device,
            logger=logger
        )
        
        # Train model
        results = train_model_with_enhanced_trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            verbose=args.verbose,
            logger=logger
        )
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model saved to {args.checkpoint_dir}")
        logger.info(f"Tokenizer saved to {args.tokenizer_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def retrain(args, logger):
    """
    Continue training an existing model.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("STARTING RETRAINING MODE")
    logger.info("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load and split data
        train_texts, val_texts, test_texts = load_and_split_data(
            data_path=args.data_path,
            split_method=args.split_method,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split_regex=args.split_regex,
            seed=args.seed,
            logger=logger
        )
        
        # Load existing model and tokenizer
        logger.info(f"Loading existing model from {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=device
        )
        
        # Update model's verbose flag
        if hasattr(model, 'config'):
            model.config['verbose'] = args.verbose
        
        # Prepare data for training
        train_dataloader, val_dataloader = prepare_data_for_training(
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
            device=device,
            logger=logger
        )
        
        # Create new checkpoint directory for continued training
        continued_checkpoint_dir = os.path.join(args.checkpoint_dir, "continued")
        os.makedirs(continued_checkpoint_dir, exist_ok=True)
        
        # Train model
        results = train_model_with_enhanced_trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            checkpoint_dir=continued_checkpoint_dir,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            verbose=args.verbose,
            logger=logger
        )
        
        logger.info("=" * 80)
        logger.info("RETRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model saved to {continued_checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def test(args, logger):
    """
    Test a trained model.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("STARTING TESTING MODE")
    logger.info("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load test data
        _, _, test_texts = load_and_split_data(
            data_path=args.data_path,
            split_method=args.split_method,
            val_ratio=0,  # No validation split needed for testing
            test_ratio=1.0,  # Use all data for testing
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split_regex=args.split_regex,
            seed=args.seed,
            logger=logger
        )
        
        # Load model and tokenizer
        logger.info(f"Loading model from {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=device
        )
        
        # Update model's verbose flag
        if hasattr(model, 'config'):
            model.config['verbose'] = args.verbose
        
        # Create model tester
        tester = ModelTester(model, tokenizer, device=device)
        
        # Test model
        logger.info("Starting model testing")
        
        # Calculate perplexity
        logger.info("Calculating perplexity")
        perplexity = tester.calculate_perplexity(test_texts)
        logger.info(f"Perplexity: {perplexity:.2f}")
        
        # Generate samples for qualitative evaluation
        logger.info("Generating samples for qualitative evaluation")
        num_samples = min(5, len(test_texts))
        
        samples = []
        for i in range(num_samples):
            # Use the first 10 tokens as prompt
            prompt_text = test_texts[i][:50]  # Take first 50 chars as prompt
            
            # Generate continuation
            generated_text = tester.generate(
                prompt_text,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            samples.append({
                "prompt": prompt_text,
                "generated": generated_text,
                "reference": test_texts[i][50:50+len(generated_text)]  # Same length as generated
            })
        
        # Save test results
        os.makedirs(args.test_output_dir, exist_ok=True)
        results = {
            "perplexity": perplexity,
            "samples": samples,
            "model_path": args.model_path,
            "data_path": args.data_path,
            "test_size": len(test_texts)
        }
        
        with open(os.path.join(args.test_output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Print samples
        logger.info("\nSample generations:")
        for i, sample in enumerate(samples):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Prompt: {sample['prompt']}")
            logger.info(f"Generated: {sample['generated']}")
            logger.info(f"Reference: {sample['reference']}")
        
        logger.info("=" * 80)
        logger.info("TESTING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Test results saved to {os.path.join(args.test_output_dir, 'test_results.json')}")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def generate(args, logger):
    """
    Generate text using a trained model.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("STARTING GENERATION MODE")
    logger.info("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model from {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=device
        )
        
        # Update model's verbose flag
        if hasattr(model, 'config'):
            model.config['verbose'] = args.verbose
        
        # Create model tester for generation
        tester = ModelTester(model, tokenizer, device=device)
        
        # Get prompt
        prompt = args.prompt
        if not prompt:
            logger.info("No prompt provided. Please enter a prompt:")
            prompt = input("> ")
        
        # Generate text
        logger.info(f"Generating text with prompt: '{prompt}'")
        logger.info(f"Parameters: max_length={args.max_length}, temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        
        start_time = time.time()
        generated_text = tester.generate(
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        end_time = time.time()
        
        generation_time = end_time - start_time
        tokens_per_second = args.max_length / generation_time
        
        # Print generated text
        logger.info("\nGenerated text:")
        logger.info("-" * 40)
        logger.info(generated_text)
        logger.info("-" * 40)
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec)")
        
        # Save generated text
        os.makedirs("./generations", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f"./generations/generation_{timestamp}.txt", "w") as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write(generated_text)
        
        logger.info("=" * 80)
        logger.info("GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Generated text saved to ./generations/generation_{timestamp}.txt")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file if args.log_file else None)
    
    # Print banner
    logger.info("=" * 80)
    logger.info("MINI LLM PROJECT - UNIFIED CLI INTERFACE")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {get_device(args.device)}")
    logger.info(f"Verbose logging: {args.verbose}")
    logger.info("=" * 80)
    
    # Run the appropriate mode
    if args.mode == "train":
        train(args, logger)
    elif args.mode == "retrain":
        retrain(args, logger)
    elif args.mode == "test":
        test(args, logger)
    elif args.mode == "generate":
        generate(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
