"""
Enhanced main script for training and using the mini LLM model.
Supports multiple file formats, model types, and detailed logging.
"""
import os
import argparse
import torch
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import tokenizers
from tokenizers import CharacterTokenizer, BPETokenizer

# Import models
from models import TransformerModel, DecoderOnlyTransformer, EncoderOnlyModel

# Import training utilities
from training import train_model
from training.model_extraction import continue_training

# Import testing utilities
from testing import ModelTester

# Import utilities
from utils import set_seed, save_json, load_json
from utils.file_loader import FileLoader, DataSplitter


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Mini LLM Project")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train", "retrain", "test", "generate"],
                        help="Operation mode: train, retrain, test, or generate")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data/corpus.txt",
                        help="Path to the training data file (supports .txt, .html, .pdf, .json)")
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
    
    return parser.parse_args()


def train(args, logger):
    """Train a new model."""
    logger.info("Starting training mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}")
    try:
        # Use the enhanced file loader to support multiple formats
        text_data = FileLoader.load_file(args.data_path)
        logger.info(f"Loaded {len(text_data)} characters of text data")
        
        # Split text into samples using the specified method
        logger.info(f"Splitting text using method: {args.split_method}")
        if args.split_method == "paragraphs":
            samples = DataSplitter.split_by_paragraphs(text_data)
        elif args.split_method == "sentences":
            samples = DataSplitter.split_by_sentences(text_data)
        elif args.split_method == "chunks":
            samples = DataSplitter.split_by_chunks(
                text_data, 
                chunk_size=args.chunk_size, 
                overlap=args.chunk_overlap
            )
        elif args.split_method == "headings":
            samples = DataSplitter.split_by_headings(text_data)
        elif args.split_method == "regex":
            samples = DataSplitter.split_by_regex(text_data, args.split_regex)
        
        logger.info(f"Created {len(samples)} text samples")
        
        # Split dataset into train/val/test
        train_size = int(len(samples) * (1 - args.val_ratio - args.test_ratio))
        val_size = int(len(samples) * args.val_ratio)
        
        # Shuffle samples
        import random
        random.seed(args.seed)
        random.shuffle(samples)
        
        train_texts = samples[:train_size]
        val_texts = samples[train_size:train_size + val_size]
        test_texts = samples[train_size + val_size:]
        
        logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
        
        # Save splits for later use
        splits_info = {
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
            "split_method": args.split_method,
            "data_path": args.data_path
        }
        
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        save_json(
            splits_info,
            os.path.join(os.path.dirname(args.data_path), "data_splits.json")
        )
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Create and train tokenizer
    logger.info(f"Creating {args.tokenizer_type} tokenizer with vocab size {args.vocab_size}")
    try:
        if args.tokenizer_type == "bpe":
            tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        else:
            tokenizer = CharacterTokenizer(vocab_size=args.vocab_size)
        
        logger.info("Training tokenizer on data")
        tokenizer.train(train_texts)
        
        # Save tokenizer
        os.makedirs(os.path.dirname(args.tokenizer_path), exist_ok=True)
        tokenizer.save(args.tokenizer_path)
        logger.info(f"Tokenizer saved to {args.tokenizer_path}")
        
    except Exception as e:
        logger.error(f"Error creating tokenizer: {str(e)}")
        return
    
    # Create model
    logger.info(f"Creating {args.model_type} model")
    try:
        # Prepare model config
        model_config = {
            "vocab_size": len(tokenizer.token_to_id),
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "dropout": args.dropout,
            "verbose": args.verbose  # Pass verbose flag to model for detailed logging
        }
        
        if args.model_type == "transformer":
            model = TransformerModel(model_config)
        elif args.model_type == "decoder_only":
            model = DecoderOnlyTransformer(model_config)
        elif args.model_type == "encoder_only":
            model = EncoderOnlyModel(model_config)
        
        logger.info(f"Model created with {model.get_parameter_count()['total']:,} parameters")
        
        # Save model configuration
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        save_json(
            model_config,
            os.path.join(args.checkpoint_dir, "model_config.json")
        )
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return
    
    # Train model
    logger.info("Starting model training")
    try:
        trained_model, history = train_model(
            model=model,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            verbose=args.verbose
        )
        
        # Save training history
        save_json(
            history,
            os.path.join(args.checkpoint_dir, "training_history.json")
        )
        
        logger.info(f"Training completed. Model saved to {args.model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return


def retrain(args, logger):
    """Continue training an existing model."""
    logger.info("Starting retraining mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}")
    try:
        # Use the enhanced file loader to support multiple formats
        text_data = FileLoader.load_file(args.data_path)
        logger.info(f"Loaded {len(text_data)} characters of text data")
        
        # Split text into samples using the specified method
        logger.info(f"Splitting text using method: {args.split_method}")
        if args.split_method == "paragraphs":
            samples = DataSplitter.split_by_paragraphs(text_data)
        elif args.split_method == "sentences":
            samples = DataSplitter.split_by_sentences(text_data)
        elif args.split_method == "chunks":
            samples = DataSplitter.split_by_chunks(
                text_data, 
                chunk_size=args.chunk_size, 
                overlap=args.chunk_overlap
            )
        elif args.split_method == "headings":
            samples = DataSplitter.split_by_headings(text_data)
        elif args.split_method == "regex":
            samples = DataSplitter.split_by_regex(text_data, args.split_regex)
        
        logger.info(f"Created {len(samples)} text samples")
        
        # Split dataset into train/val/test
        train_size = int(len(samples) * (1 - args.val_ratio - args.test_ratio))
        val_size = int(len(samples) * args.val_ratio)
        
        # Shuffle samples
        import random
        random.seed(args.seed)
        random.shuffle(samples)
        
        train_texts = samples[:train_size]
        val_texts = samples[train_size:train_size + val_size]
        test_texts = samples[train_size + val_size:]
        
        logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    try:
        if args.tokenizer_type == "bpe":
            tokenizer = BPETokenizer.load(args.tokenizer_path)
        else:
            tokenizer = CharacterTokenizer.load(args.tokenizer_path)
        logger.info(f"Tokenizer loaded with vocabulary size {len(tokenizer.token_to_id)}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        return
    
    # Continue training the model
    logger.info(f"Continuing training for model at {args.model_path}")
    try:
        # Use the model extraction utility to continue training
        trained_model, history = continue_training(
            model_path=args.model_path,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            verbose=args.verbose
        )
        
        # Save training history
        save_json(
            history,
            os.path.join(args.checkpoint_dir, "retraining_history.json")
        )
        
        logger.info(f"Retraining completed. Model saved to {args.model_path}")
        
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return


def test(args, logger):
    """Test a trained model."""
    logger.info("Starting testing mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    try:
        if args.tokenizer_type == "bpe":
            tokenizer = BPETokenizer.load(args.tokenizer_path)
        else:
            tokenizer = CharacterTokenizer.load(args.tokenizer_path)
        logger.info(f"Tokenizer loaded with vocabulary size {len(tokenizer.token_to_id)}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        return
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    try:
        # Load model config
        config_path = f"{args.model_path}_config.json"
        if not os.path.exists(config_path):
            # Try alternative path
            config_path = os.path.join(os.path.dirname(args.model_path), "model_config.json")
            if not os.path.exists(config_path):
                logger.error(f"Model config file not found at {args.model_path}_config.json or in checkpoint directory")
                return
            
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        
        # Add verbose flag to config
        config["verbose"] = args.verbose
        
        # Create model instance based on config or specified type
        model_type = config.get("model_type", args.model_type)
        
        if model_type == "transformer":
            model = TransformerModel(config)
        elif model_type == "decoder_only":
            model = DecoderOnlyTransformer(config)
        elif model_type == "encoder_only":
            model = EncoderOnlyModel(config)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return
        
        # Load model weights with enhanced error handling
        checkpoint_path = f"{args.model_path}.pt"
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint file not found: {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check for nested state dict
            if "model_state_dict" in checkpoint:
                logger.info("Loading model from nested state_dict")
                model.load_state_dict(checkpoint["model_state_dict"])
            # Check for direct state dict (keys match model parameters)
            elif any(key in checkpoint for key in model.state_dict().keys()):
                logger.info("Loading model from direct state_dict")
                # Filter out any non-parameter keys
                filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
                model.load_state_dict(filtered_state_dict, strict=False)
                if len(filtered_state_dict) < len(model.state_dict()):
                    logger.warning(f"Loaded a subset of parameters ({len(filtered_state_dict)}/{len(model.state_dict())})")
            else:
                # Try loading directly, might fail but worth a try
                logger.info("Attempting to load model directly")
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    logger.error(f"Failed to load model state: {str(e)}")
                    logger.error(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                    logger.error(f"Model expects keys like: {list(model.state_dict().keys())[:5]}")
                    return
        else:
            # Direct state dict
            logger.info("Loading model from direct state_dict")
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded with {model.get_parameter_count()['total']:,} parameters")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    try:
        # Use the enhanced file loader to support multiple formats
        text_data = FileLoader.load_file(args.data_path)
        
        # Split text into samples using the specified method
        if args.split_method == "paragraphs":
            samples = DataSplitter.split_by_paragraphs(text_data)
        elif args.split_method == "sentences":
            samples = DataSplitter.split_by_sentences(text_data)
        elif args.split_method == "chunks":
            samples = DataSplitter.split_by_chunks(
                text_data, 
                chunk_size=args.chunk_size, 
                overlap=args.chunk_overlap
            )
        elif args.split_method == "headings":
            samples = DataSplitter.split_by_headings(text_data)
        elif args.split_method == "regex":
            samples = DataSplitter.split_by_regex(text_data, args.split_regex)
        
        # Use a portion for testing
        test_size = min(100, len(samples))  # Limit to 100 samples for efficiency
        test_texts = samples[:test_size]
        
        logger.info(f"Using {len(test_texts)} samples for testing")
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Create tester
    logger.info("Creating model tester")
    tester = ModelTester(model, tokenizer, device)
    
    # Create prompt-completion pairs for accuracy testing
    # For simplicity, we'll use the first half of each test sample as prompt
    # and the second half as expected completion
    prompt_completion_pairs = []
    for text in test_texts:
        if len(text) < 20:  # Skip very short samples
            continue
        
        mid_point = len(text) // 2
        prompt = text[:mid_point]
        completion = text[mid_point:]
        
        prompt_completion_pairs.append((prompt, completion))
    
    # Run comprehensive test
    logger.info("Running comprehensive model testing")
    try:
        results = tester.run_comprehensive_test(
            test_texts=test_texts,
            prompt_completion_pairs=prompt_completion_pairs,
            output_dir=args.test_output_dir,
            max_length=args.max_length
        )
        
        logger.info(f"Testing completed. Results saved to {args.test_output_dir}")
        logger.info(f"Perplexity: {results['perplexity']:.2f}")
        logger.info(f"Accuracy: {results['accuracy']}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return


def generate(args, logger):
    """Generate text using a trained model."""
    logger.info("Starting text generation mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    try:
        if args.tokenizer_type == "bpe":
            tokenizer = BPETokenizer.load(args.tokenizer_path)
        else:
            tokenizer = CharacterTokenizer.load(args.tokenizer_path)
        logger.info(f"Tokenizer loaded with vocabulary size {len(tokenizer.token_to_id)}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        return
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    try:
        # Load model config
        config_path = f"{args.model_path}_config.json"
        if not os.path.exists(config_path):
            # Try alternative path
            config_path = os.path.join(os.path.dirname(args.model_path), "model_config.json")
            if not os.path.exists(config_path):
                logger.error(f"Model config file not found at {args.model_path}_config.json or in checkpoint directory")
                return
            
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        
        # Add verbose flag to config
        config["verbose"] = args.verbose
        
        # Create model instance based on config or specified type
        model_type = config.get("model_type", args.model_type)
        
        if model_type == "transformer":
            model = TransformerModel(config)
        elif model_type == "decoder_only":
            model = DecoderOnlyTransformer(config)
        elif model_type == "encoder_only":
            model = EncoderOnlyModel(config)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return
        
        # Load model weights with enhanced error handling
        checkpoint_path = f"{args.model_path}.pt"
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint file not found: {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check for nested state dict
            if "model_state_dict" in checkpoint:
                logger.info("Loading model from nested state_dict")
                model.load_state_dict(checkpoint["model_state_dict"])
            # Check for direct state dict (keys match model parameters)
            elif any(key in checkpoint for key in model.state_dict().keys()):
                logger.info("Loading model from direct state_dict")
                # Filter out any non-parameter keys
                filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
                model.load_state_dict(filtered_state_dict, strict=False)
                if len(filtered_state_dict) < len(model.state_dict()):
                    logger.warning(f"Loaded a subset of parameters ({len(filtered_state_dict)}/{len(model.state_dict())})")
            else:
                # Try loading directly, might fail but worth a try
                logger.info("Attempting to load model directly")
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    logger.error(f"Failed to load model state: {str(e)}")
                    logger.error(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                    logger.error(f"Model expects keys like: {list(model.state_dict().keys())[:5]}")
                    return
        else:
            # Direct state dict
            logger.info("Loading model from direct state_dict")
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded with {model.get_parameter_count()['total']:,} parameters")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Get prompt
    prompt = args.prompt
    if not prompt:
        logger.info("No prompt provided, using default prompt")
        prompt = "Once upon a time"
    
    logger.info(f"Generating text with prompt: '{prompt}'")
    
    # Generate text
    try:
        start_time = time.time()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt)
        
        # Generate text
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        # Decode output
        generated_text = tokenizer.decode(output_ids)
        
        end_time = time.time()
        
        logger.info(f"Text generation completed in {end_time - start_time:.2f} seconds")
        
        # Print generated text
        print("\n" + "="*50)
        print("GENERATED TEXT:")
        print("="*50)
        print(generated_text)
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(verbose=args.verbose)
    
    logger.info(f"Running in {args.mode} mode")
    
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


if __name__ == "__main__":
    main()
