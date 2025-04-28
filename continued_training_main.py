"""
Enhanced main script with continued training functionality.
"""
import argparse
import logging
import os
import torch
import json
from pathlib import Path

from tokenizers import BPETokenizer, CharacterTokenizer
from models import TransformerModel, DecoderOnlyTransformer
from training.enhanced_data_utils import load_and_split_data, load_multiple_files, create_dataloaders
from training.trainer import Trainer
from testing.model_tester import ModelTester
from utils.helpers import setup_logging


def main():
    """Main entry point for the Mini LLM project."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Mini LLM Project")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["train", "continue_training", "test", "generate", "ui"],
                        help="Mode to run: train, continue_training, test, generate, or ui")
    
    # Data loading options
    parser.add_argument("--data_path", type=str, help="Path to data file or directory")
    parser.add_argument("--file_type", type=str, choices=["txt", "pdf", "html", "auto"], default="auto",
                        help="Type of file to load (auto detects from extension)")
    parser.add_argument("--split_method", type=str, 
                        choices=["paragraph", "sentence", "chunk", "semantic", "regex"], 
                        default="paragraph",
                        help="Method to split text into samples")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Minimum length of a text sample (characters)")
    parser.add_argument("--max_length", type=int, default=10000,
                        help="Maximum length of a text sample (characters)")
    
    # Model and tokenizer options
    parser.add_argument("--model_type", type=str, choices=["transformer", "decoder_only"], default="decoder_only",
                        help="Type of model to use")
    parser.add_argument("--tokenizer_type", type=str, choices=["bpe", "character"], default="bpe",
                        help="Type of tokenizer to use")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size for tokenizer")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training options
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    # Model loading options
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model",
                        help="Path to model checkpoint (without extension)")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizer.json",
                        help="Path to tokenizer file")
    
    # Generation options
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt for text generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k for sampling")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Mini LLM Project - Mode: {args.mode}")
    
    # Create directories if they don't exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Run selected mode
    if args.mode == "train":
        run_training(args, device, logger)
    elif args.mode == "continue_training":
        run_continued_training(args, device, logger)
    elif args.mode == "test":
        run_testing(args, device, logger)
    elif args.mode == "generate":
        run_generation(args, device, logger)
    elif args.mode == "ui":
        run_ui(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")


def run_training(args, device, logger):
    """Run training mode."""
    logger.info("Starting training mode")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    
    # Check if data_path is a directory or a file
    data_path = Path(args.data_path)
    if data_path.is_dir():
        # Load multiple files from directory
        file_paths = [str(f) for f in data_path.glob("*") if f.is_file()]
        train_texts, val_texts, test_texts = load_multiple_files(
            file_paths=file_paths,
            split_method=args.split_method,
            min_length=args.min_length,
            max_length=args.max_length,
            val_split=args.val_split,
            test_split=args.test_split
        )
    else:
        # Load single file
        train_texts, val_texts, test_texts = load_and_split_data(
            file_path=args.data_path,
            split_method=args.split_method,
            min_length=args.min_length,
            max_length=args.max_length,
            val_split=args.val_split,
            test_split=args.test_split
        )
    
    logger.info(f"Loaded {len(train_texts) + len(val_texts) + len(test_texts)} text samples")
    logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    # Create tokenizer
    tokenizer_path = Path("./data/tokenizer.json")
    
    if args.tokenizer_type == "bpe":
        logger.info(f"Creating bpe tokenizer with vocab size {args.vocab_size}")
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    else:
        logger.info("Creating character tokenizer")
        tokenizer = CharacterTokenizer()
    
    # Train tokenizer on data
    logger.info("Training tokenizer on data")
    tokenizer.train(train_texts)
    
    # Save tokenizer
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Create model
    if args.model_type == "transformer":
        logger.info("Creating transformer model")
        model = TransformerModel({
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "dropout": args.dropout
        })
    else:
        logger.info("Creating decoder_only model")
        model = DecoderOnlyTransformer({
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "dropout": args.dropout
        })
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    logger.info("Starting model training")
    trainer.train(epochs=args.epochs)
    
    logger.info("Training complete")


def run_continued_training(args, device, logger):
    """Run continued training mode (fine-tuning an existing model)."""
    logger.info("Starting continued training mode")
    
    # Check if model exists
    model_path = args.model_path
    config_path = f"{model_path}_config.json"
    model_file = f"{model_path}.pt"
    
    if not os.path.exists(config_path) or not os.path.exists(model_file):
        logger.error(f"Model files not found: {config_path} or {model_file}")
        logger.error("Cannot continue training without an existing model")
        return
    
    # Load model configuration
    logger.info(f"Loading model configuration from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create model with the same architecture
    if args.model_type == "transformer":
        logger.info("Loading transformer model")
        model = TransformerModel(config)
    else:
        logger.info("Loading decoder_only model")
        model = DecoderOnlyTransformer(config)
    
    # Load model weights
    logger.info(f"Loading model weights from {model_file}")
    checkpoint = torch.load(model_file, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    if args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = CharacterTokenizer.load(args.tokenizer_path)
    
    # Load new training data
    logger.info(f"Loading new training data from {args.data_path}")
    
    # Check if data_path is a directory or a file
    data_path = Path(args.data_path)
    if data_path.is_dir():
        # Load multiple files from directory
        file_paths = [str(f) for f in data_path.glob("*") if f.is_file()]
        train_texts, val_texts, test_texts = load_multiple_files(
            file_paths=file_paths,
            split_method=args.split_method,
            min_length=args.min_length,
            max_length=args.max_length,
            val_split=args.val_split,
            test_split=args.test_split
        )
    else:
        # Load single file
        train_texts, val_texts, test_texts = load_and_split_data(
            file_path=args.data_path,
            split_method=args.split_method,
            min_length=args.min_length,
            max_length=args.max_length,
            val_split=args.val_split,
            test_split=args.test_split
        )
    
    logger.info(f"Loaded {len(train_texts) + len(val_texts) + len(test_texts)} new text samples")
    logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_len
    )
    
    # Create trainer with loaded model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate * 0.5,  # Lower learning rate for fine-tuning
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Continue training
    logger.info("Starting continued training")
    trainer.train(epochs=args.epochs)
    
    logger.info("Continued training complete")


def run_testing(args, device, logger):
    """Run testing mode."""
    logger.info("Starting testing mode")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    if args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = CharacterTokenizer.load(args.tokenizer_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    config_path = f"{args.model_path}_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if args.model_type == "transformer":
        model = TransformerModel(config)
    else:
        model = DecoderOnlyTransformer(config)
    
    checkpoint = torch.load(f"{args.model_path}.pt", map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load test data if provided
    if args.data_path:
        logger.info(f"Loading test data from {args.data_path}")
        
        # Check if data_path is a directory or a file
        data_path = Path(args.data_path)
        if data_path.is_dir():
            # Load multiple files from directory
            file_paths = [str(f) for f in data_path.glob("*") if f.is_file()]
            _, _, test_texts = load_multiple_files(
                file_paths=file_paths,
                split_method=args.split_method,
                min_length=args.min_length,
                max_length=args.max_length,
                val_split=0,
                test_split=1.0  # Use all data for testing
            )
        else:
            # Load single file
            _, _, test_texts = load_and_split_data(
                file_path=args.data_path,
                split_method=args.split_method,
                min_length=args.min_length,
                max_length=args.max_length,
                val_split=0,
                test_split=1.0  # Use all data for testing
            )
        
        logger.info(f"Loaded {len(test_texts)} test samples")
        
        # Create test dataloader
        from torch.utils.data import DataLoader
        from training.data_utils import TextDataset
        
        test_dataset = TextDataset(
            texts=test_texts,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            is_training=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create tester
        tester = ModelTester(
            model=model,
            test_dataloader=test_dataloader,
            device=device
        )
        
        # Test model
        logger.info("Starting model testing")
        metrics = tester.evaluate()
        
        logger.info(f"Test loss: {metrics['loss']:.4f}")
        logger.info(f"Test perplexity: {metrics['perplexity']:.2f}")
    
    else:
        logger.warning("No test data provided. Skipping testing.")
    
    logger.info("Testing complete")


def run_generation(args, device, logger):
    """Run text generation mode."""
    logger.info("Starting generation mode")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    if args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = CharacterTokenizer.load(args.tokenizer_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    config_path = f"{args.model_path}_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if args.model_type == "transformer":
        model = TransformerModel(config)
    else:
        model = DecoderOnlyTransformer(config)
    
    checkpoint = torch.load(f"{args.model_path}.pt", map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Generate text
    if args.prompt:
        logger.info(f"Generating text from prompt: {args.prompt}")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(args.prompt)
        
        # Add BOS token if not present
        if input_ids[0] != tokenizer.token_to_id["[BOS]"]:
            input_ids = [tokenizer.token_to_id["[BOS]"]] + input_ids
        
        # Convert to tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=args.max_seq_len,
                temperature=args.temperature,
                top_k=args.top_k
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        print("\nGenerated Text:")
        print("-" * 40)
        print(generated_text)
        print("-" * 40)
    
    else:
        logger.warning("No prompt provided. Skipping generation.")
    
    logger.info("Generation complete")


def run_ui(args, logger):
    """Run UI mode."""
    logger.info("Starting UI mode")
    
    try:
        import streamlit
        logger.info("Launching Streamlit UI")
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if rewritten_app.py exists
        rewritten_app_path = os.path.join(script_dir, "ui", "rewritten_app.py")
        if os.path.exists(rewritten_app_path):
            os.system(f"streamlit run {rewritten_app_path}")
        else:
            # Fall back to regular app.py
            app_path = os.path.join(script_dir, "ui", "app.py")
            os.system(f"streamlit run {app_path}")
        
    except ImportError:
        logger.error("Streamlit is not installed. Please install it with: pip install streamlit")


if __name__ == "__main__":
    main()
