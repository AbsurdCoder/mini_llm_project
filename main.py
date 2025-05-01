"""
Unified main script for the Mini LLM project.
Provides a comprehensive CLI interface for all model operations.

Features:
- Multi-format data loading (.txt, .html, .pdf, .json)
- Multiple model architectures (Transformer, Encoder-only, Decoder-only)
- Transfer learning with Hugging Face pre-trained models
- Detailed logging of model internals and training progress
- Model training, retraining, testing, and generation
- Comprehensive error handling and validation

Usage:
    # Custom model training
    python main.py --mode train --data_path ./data/corpus.txt
    
    # Transfer learning (fine-tuning)
    python main.py --mode train --hf_model_name distilbert/distilbert-base-uncased --data_path ./data/corpus.txt
    
    # Retraining custom model
    python main.py --mode retrain --model_path ./checkpoints/best_model --data_path ./data/new_corpus.txt
    
    # Testing custom model
    python main.py --mode test --model_path ./checkpoints/best_model --data_path ./data/test_corpus.txt
    
    # Testing HF model (after fine-tuning)
    python main.py --mode test --model_path ./checkpoints/hf_finetuned/best_model --hf_model_name distilbert/distilbert-base-uncased --data_path ./data/test_corpus.txt
    
    # Generating with custom model
    python main.py --mode generate --model_path ./checkpoints/best_model --prompt "Once upon a time"
    
    # Generating with HF model
    python main.py --mode generate --model_path ./checkpoints/hf_finetuned/best_model --hf_model_name distilbert/distilbert-base-uncased --prompt "Once upon a time"
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
from ctokenizers.bpe_tokenizer import BPETokenizer
from ctokenizers.character_tokenizer import CharacterTokenizer

# Import models
from models.transformer_model import TransformerModel, DecoderOnlyTransformer
from models.encoder_model import EncoderOnlyModel
from models.enhanced_transformer_components import (
    MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding,
    TransformerEncoderLayer, TransformerDecoderLayer
)

# Import training utilities
from training.enhanced_detailed_trainer import EnhancedTrainer
from training.model_extraction import load_model_and_tokenizer, continue_training

# Import testing utilities
from testing.model_tester import ModelTester

# Import utilities
from utils.helpers import set_seed
from utils.file_loader import FileLoader, DataSplitter

# Try importing Hugging Face transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig = None, None, None, None

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
        description="Mini LLM Project - Unified CLI Interface with Transfer Learning",
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
    
    # Tokenizer arguments (Custom)
    parser.add_argument("--tokenizer_type", type=str, default="bpe", 
                        choices=["bpe", "character"],
                        help="Type of custom tokenizer to use (ignored if --hf_model_name is set)")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size for the custom tokenizer (ignored if --hf_model_name is set)")
    parser.add_argument("--tokenizer_path", type=str, default="../data/tokenizer.json",
                        help="Path to save/load the custom tokenizer (ignored if --hf_model_name is set)")
    
    # Model arguments (Custom)
    parser.add_argument("--model_type", type=str, default="decoder_only", 
                        choices=["transformer", "decoder_only", "encoder_only"],
                        help="Type of custom model to use (ignored if --hf_model_name is set)")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Custom model dimension (ignored if --hf_model_name is set)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads for custom model (ignored if --hf_model_name is set)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers for custom model (ignored if --hf_model_name is set)")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension for custom model (ignored if --hf_model_name is set)")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length (used for both custom and HF models)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for custom model (ignored if --hf_model_name is set)")

    # Hugging Face Transfer Learning Arguments
    parser.add_argument("--hf_model_name", type=str, default="",
                        help="Name or path of the Hugging Face pre-trained model to use (e.g., 'distilbert/distilbert-base-uncased', 'openai-community/gpt2'). If set, custom model/tokenizer args are ignored.")
    parser.add_argument("--hf_tokenizer_name", type=str, default="",
                        help="Optional: Name or path of the Hugging Face tokenizer (defaults to hf_model_name if not set)")
    parser.add_argument("--offline_hf_dir", type=str, default="",
                        help="Optional: Path to a local directory containing downloaded Hugging Face model/tokenizer files for offline use.")

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
                        help="Maximum length for generation (including prompt)")
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
                        help="Path to save/load the model checkpoint (base name without extension)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging (including model internals for custom models)")
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to log file (if empty, logs to console only)")
    
    args = parser.parse_args()
    
    # Validation for Hugging Face mode
    if args.hf_model_name and not HUGGINGFACE_AVAILABLE:
        parser.error("Hugging Face 'transformers' library not found. Please install it (`pip install transformers`) to use --hf_model_name.")
        
    # Set default HF tokenizer name if not provided
    if args.hf_model_name and not args.hf_tokenizer_name:
        args.hf_tokenizer_name = args.hf_model_name
        
    # Ensure model_path directory exists if saving
    if args.mode in ["train", "retrain"]:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
    return args


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
        else:
            # Default to paragraph splitting if method is unknown
            logger.warning(f"Unknown split method '{split_method}', defaulting to 'paragraphs'.")
            samples = DataSplitter.split_by_paragraphs(text_data)
        
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
    args: argparse.Namespace,
    train_texts: List[str],
    logger: logging.Logger
) -> Union[BPETokenizer, CharacterTokenizer, AutoTokenizer]:
    """
    Create or load the appropriate tokenizer.
    
    Args:
        args: Command line arguments
        train_texts: Training texts for tokenizer training (if needed)
        logger: Logger instance
        
    Returns:
        Initialized tokenizer instance
    """
    if args.hf_model_name:
        # Use Hugging Face tokenizer
        logger.info(f"Loading Hugging Face tokenizer: {args.hf_tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.hf_tokenizer_name,
                cache_dir=args.offline_hf_dir if args.offline_hf_dir else None,
                local_files_only=bool(args.offline_hf_dir)
            )
            # Ensure padding token is set for HF tokenizers
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
                else:
                    # Add a default pad token if none exists
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added default pad_token: [PAD]")
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading Hugging Face tokenizer: {str(e)}")
            raise
    else:
        # Use custom tokenizer
        tokenizer_path = Path(args.tokenizer_path)
        if tokenizer_path.exists() and args.mode != "train":
            logger.info(f"Loading existing custom tokenizer from {tokenizer_path}")
            try:
                if args.tokenizer_type == "bpe":
                    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
                    tokenizer.load(str(tokenizer_path))
                elif args.tokenizer_type == "character":
                    tokenizer = CharacterTokenizer()
                    tokenizer.load(str(tokenizer_path))
                else:
                    raise ValueError(f"Unsupported custom tokenizer type: {args.tokenizer_type}")
                return tokenizer
            except Exception as e:
                logger.error(f"Error loading custom tokenizer: {str(e)}")
                raise
        else:
            logger.info(f"Creating and training a new custom {args.tokenizer_type} tokenizer")
            try:
                if args.tokenizer_type == "bpe":
                    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
                    tokenizer.train(train_texts)
                elif args.tokenizer_type == "character":
                    tokenizer = CharacterTokenizer()
                    tokenizer.train(train_texts)
                else:
                    raise ValueError(f"Unsupported custom tokenizer type: {args.tokenizer_type}")
                
                # Save the trained tokenizer
                tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
                tokenizer.save(str(tokenizer_path))
                logger.info(f"Saved new custom tokenizer to {tokenizer_path}")
                return tokenizer
            except Exception as e:
                logger.error(f"Error training/saving custom tokenizer: {str(e)}")
                raise


def create_model(
    args: argparse.Namespace,
    tokenizer: Union[BPETokenizer, CharacterTokenizer, AutoTokenizer],
    config: Optional[Dict[str, Any]] = None,
    logger: logging.Logger = None
) -> Union[torch.nn.Module, AutoModelForCausalLM, AutoModelForMaskedLM]:
    """
    Create the model based on arguments or loaded config.
    
    Args:
        args: Command line arguments
        tokenizer: Initialized tokenizer
        config: Optional loaded model configuration
        logger: Logger instance
        
    Returns:
        Initialized model instance
    """
    if args.hf_model_name:
        # Load Hugging Face model
        logger.info(f"Loading Hugging Face model: {args.hf_model_name}")
        try:
            # Determine model type (causal or masked)
            # This is a heuristic, might need refinement
            if "gpt" in args.hf_model_name.lower() or "causal" in args.hf_model_name.lower():
                model_class = AutoModelForCausalLM
                logger.info("Assuming Causal LM model type.")
            else:
                model_class = AutoModelForMaskedLM
                logger.info("Assuming Masked LM model type.")
                
            model = model_class.from_pretrained(
                args.hf_model_name,
                cache_dir=args.offline_hf_dir if args.offline_hf_dir else None,
                local_files_only=bool(args.offline_hf_dir)
            )
            # Resize token embeddings if tokenizer vocab size changed (e.g., added pad token)
            model.resize_token_embeddings(len(tokenizer))
            return model
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {str(e)}")
            raise
    else:
        # Create custom model
        if config:
            logger.info("Creating custom model from loaded configuration")
            vocab_size = config.get("vocab_size", args.vocab_size)
            d_model = config.get("d_model", args.d_model)
            num_heads = config.get("num_heads", args.num_heads)
            num_layers = config.get("num_layers", args.num_layers)
            d_ff = config.get("d_ff", args.d_ff)
            max_seq_len = config.get("max_seq_len", args.max_seq_len)
            dropout = config.get("dropout", args.dropout)
            model_type = config.get("model_type", args.model_type)
        else:
            logger.info("Creating new custom model from command line arguments")
            vocab_size = tokenizer.get_vocab_size()
            d_model = args.d_model
            num_heads = args.num_heads
            num_layers = args.num_layers
            d_ff = args.d_ff
            max_seq_len = args.max_seq_len
            dropout = args.dropout
            model_type = args.model_type
            
        logger.info(f"Custom model parameters: type={model_type}, vocab_size={vocab_size}, d_model={d_model}, heads={num_heads}, layers={num_layers}, d_ff={d_ff}, max_seq_len={max_seq_len}, dropout={dropout}")
        
        try:
            # Create config dictionary
            model_config = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "dropout": dropout,
                "model_type": model_type
            }
            
            if model_type == "transformer":
                model = TransformerModel(config=model_config)
            elif model_type == "decoder_only":
                model = DecoderOnlyTransformer(config=model_config)
            elif model_type == "encoder_only":
                model = EncoderOnlyModel(config=model_config)
            else:
                raise ValueError(f"Unsupported custom model type: {model_type}")
            return model
        except Exception as e:
            logger.error(f"Error creating custom model: {str(e)}")
            raise


def train_new_model(
    args: argparse.Namespace,
    device: str,
    logger: logging.Logger
):
    """
    Train a new model from scratch or fine-tune a pre-trained HF model.
    """
    logger.info("Starting new model training...")
    
    # 1. Load and split data
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
    
    # 2. Create or load tokenizer
    tokenizer = create_or_load_tokenizer(args, train_texts, logger)
    
    # 3. Create model
    model = create_model(args, tokenizer, logger=logger)
    model.to(device)
    
    # 4. Prepare configuration dictionary
    if args.hf_model_name:
        # For HF models, use their config, add our custom flags
        config = model.config.to_dict() if hasattr(model.config, 'to_dict') else vars(model.config)
        config["is_hf_model"] = True
        config["hf_model_name"] = args.hf_model_name
        config["hf_tokenizer_name"] = args.hf_tokenizer_name
        config["max_seq_len"] = args.max_seq_len # Ensure max_seq_len is in config
    else:
        # For custom models, create config from args
        config = {
            "model_type": args.model_type,
            "vocab_size": tokenizer.get_vocab_size(),
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "dropout": args.dropout,
            "tokenizer_type": args.tokenizer_type,
            "is_hf_model": False
        }
        # Attach config to model for consistency
        model.config = config 
    print('2384673278146327846329874678231461328974631287946132987463218746329841632987')
    # 5. Initialize Trainer
    trainer = EnhancedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts, # Pass train_texts directly
        val_texts=val_texts,     # Pass val_texts directly
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_save_path=args.model_path, # Corrected parameter name
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        max_seq_len=args.max_seq_len, # Pass max_seq_len
        verbose=args.verbose,
        is_hf_model=config["is_hf_model"]
    )
    
    # 6. Train the model
    logger.info("Starting training process...")
    trainer.train()
    logger.info("Training finished.")
    
    # 7. Save final model and config explicitly (trainer saves best)
    final_model_path = f"{args.model_path}_final.pth"
    final_config_path = f"{args.model_path}_final_config.json"
    torch.save(model.state_dict(), final_model_path)
    with open(final_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved final model state to {final_model_path}")
    logger.info(f"Saved final model config to {final_config_path}")


def retrain_existing_model(
    args: argparse.Namespace,
    device: str,
    logger: logging.Logger
):
    """
    Continue training an existing model checkpoint.
    """
    logger.info(f"Starting model retraining from {args.model_path}")
    
    try:
        # Load model and tokenizer
        model, tokenizer, config, is_hf_model = load_model_and_tokenizer(
            model_path=args.model_path, # Use model_path argument
            tokenizer_path=args.tokenizer_path, # May be ignored if HF
            hf_model_name=args.hf_model_name, # Pass HF info
            hf_tokenizer_name=args.hf_tokenizer_name,
            offline_hf_dir=args.offline_hf_dir,
            device=device,
            logger=logger
        )
        
        # Use the returned is_hf_model flag
        is_hf = is_hf_model
        
        # Ensure max_seq_len is consistent
        loaded_max_seq_len = config.get("max_seq_len", 512) # Default if missing
        if loaded_max_seq_len != args.max_seq_len:
            logger.warning(f"Mismatch in max_seq_len: Loaded model has {loaded_max_seq_len}, but args specify {args.max_seq_len}. Using loaded value: {loaded_max_seq_len}.")
            args.max_seq_len = loaded_max_seq_len
        else:
            logger.info(f"Using max_seq_len: {args.max_seq_len}")
            
        # Load new data for retraining
        logger.info("Loading new data for retraining...")
        train_texts, val_texts, _ = load_and_split_data(
            data_path=args.data_path,
            split_method=args.split_method,
            val_ratio=args.val_ratio,
            test_ratio=0, # No test set needed for retraining phase
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split_regex=args.split_regex,
            seed=args.seed,
            logger=logger
        )
        
        # Continue training using the dedicated function
        continue_training(
            model=model,
            tokenizer=tokenizer,
            config=config,
            train_texts=train_texts,
            val_texts=val_texts,
            args=args, # Pass all args
            device=device,
            logger=logger
        )
        logger.info("Retraining finished.")
        
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def test_model(
    args: argparse.Namespace,
    device: str,
    logger: logging.Logger
):
    """
    Test a trained model on a test dataset.
    """
    logger.info(f"Starting model testing for {args.model_path}")
    
    try:
        # Load model and tokenizer
        model, tokenizer, config, is_hf_model = load_model_and_tokenizer(
            model_path=args.model_path, # Use model_path argument
            tokenizer_path=args.tokenizer_path, # May be ignored if HF
            hf_model_name=args.hf_model_name,
            hf_tokenizer_name=args.hf_tokenizer_name,
            offline_hf_dir=args.offline_hf_dir,
            device=device,
            logger=logger
            # Removed for_inference=True, handled internally now
        )
        model.eval()
        
        # Use the returned is_hf_model flag
        is_hf = is_hf_model
        
        # Ensure max_seq_len is consistent
        loaded_max_seq_len = config.get("max_seq_len", 512) # Default if missing
        if loaded_max_seq_len != args.max_seq_len:
            logger.warning(f"Mismatch in max_seq_len: Loaded model has {loaded_max_seq_len}, but args specify {args.max_seq_len}. Using loaded value: {loaded_max_seq_len}.")
            args.max_seq_len = loaded_max_seq_len
        else:
            logger.info(f"Using max_seq_len: {args.max_seq_len}")
            
        # Load test data
        logger.info("Loading test data...")
        _, _, test_texts = load_and_split_data(
            data_path=args.data_path,
            split_method=args.split_method,
            val_ratio=0, # No val set needed
            test_ratio=1.0, # Use all data as test
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split_regex=args.split_regex,
            seed=args.seed,
            logger=logger
        )
        
        if not test_texts:
            logger.error("No test data found. Cannot perform testing.")
            return
            
        # Initialize ModelTester
        tester = ModelTester(
            model=model,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
            is_hf_model=is_hf,
            max_seq_len=args.max_seq_len
        )
        
        # Run evaluation
        logger.info("Starting evaluation on test set...")
        results = tester.evaluate(test_texts, batch_size=args.batch_size)
        
        # Save results
        os.makedirs(args.test_output_dir, exist_ok=True)
        results_path = os.path.join(args.test_output_dir, f"{Path(args.model_path).stem}_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Test results saved to {results_path}")
        logger.info(f"Perplexity: {results.get('perplexity', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def generate_text(
    args: argparse.Namespace,
    device: str,
    logger: logging.Logger
):
    """
    Generate text using a trained model.
    """
    logger.info(f"Starting text generation with model {args.model_path}")
    
    if not args.prompt:
        logger.error("Prompt is required for text generation. Use --prompt 'Your text here'.")
        return
        
    try:
        # Load model and tokenizer
        model, tokenizer, config, is_hf_model = load_model_and_tokenizer(
            model_path=args.model_path, # Use model_path argument
            tokenizer_path=args.tokenizer_path, # May be ignored if HF
            hf_model_name=args.hf_model_name,
            hf_tokenizer_name=args.hf_tokenizer_name,
            offline_hf_dir=args.offline_hf_dir,
            device=device,
            logger=logger
            # Removed for_inference=True, handled internally now
        )
        model.eval()
        
        # Use the returned is_hf_model flag
        is_hf = is_hf_model
        
        # Ensure max_seq_len is consistent
        loaded_max_seq_len = config.get("max_seq_len", 512) # Default if missing
        if loaded_max_seq_len != args.max_seq_len:
            logger.warning(f"Mismatch in max_seq_len: Loaded model has {loaded_max_seq_len}, but args specify {args.max_seq_len}. Using loaded value: {loaded_max_seq_len}.")
            args.max_seq_len = loaded_max_seq_len
        else:
            logger.info(f"Using max_seq_len: {args.max_seq_len}")
            
        logger.info(f"Generating text with prompt: '{args.prompt}'")
        
        # --- Generation Logic ---
        # Tokenize prompt
        if is_hf:
            # HF Tokenizer specific encoding
            inputs = tokenizer(args.prompt, return_tensors="pt", max_length=args.max_seq_len, truncation=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
        else:
            # Custom Tokenizer encoding
            input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
            attention_mask = None # Custom models might not use attention mask in the same way during generation
            
        # Generation parameters
        gen_kwargs = {
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None,
            "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
            "do_sample": True, # Enable sampling
            "num_return_sequences": 1
        }
        # Remove None values from kwargs
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        # Add attention mask for HF models if available
        if is_hf and attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
            
        # Generate
        with torch.no_grad():
            output_sequences = model.generate(input_ids=input_ids, **gen_kwargs)
            
        # Decode generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        logger.info("--- Generated Text ---")
        print(generated_text)
        logger.info("--- End Generated Text ---")
        
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """
    Main function to orchestrate the process.
    """
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file if args.log_file else None
    logger = setup_logging(verbose=args.verbose, log_file=log_file)
    
    logger.info(f"Starting Mini LLM script in mode: {args.mode}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Determine device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    start_time = time.time()
    
    try:
        if args.mode == "train":
            train_new_model(args, device, logger)
        elif args.mode == "retrain":
            retrain_existing_model(args, device, logger)
        elif args.mode == "test":
            test_model(args, device, logger)
        elif args.mode == "generate":
            generate_text(args, device, logger)
            
        end_time = time.time()
        logger.info(f"Mode '{args.mode}' completed successfully in {end_time - start_time:.2f} seconds.")
        
    except Exception as e:
        logger.critical(f"An unhandled error occurred in mode '{args.mode}': {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    main()

