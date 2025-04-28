"""
Main script for training and using the mini LLM model.
"""
import os
import argparse
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from tokenizers import CharacterTokenizer, BPETokenizer
from models import TransformerModel, DecoderOnlyTransformer
from training import train_model
from testing import ModelTester
from utils import set_seed, load_text_file, split_dataset, save_json, load_json


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mini LLM Project")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "generate", "ui"],
                        help="Operation mode: train, test, generate, or ui")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data/corpus.txt",
                        help="Path to the training data file")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of test data")
    
    # Tokenizer arguments
    parser.add_argument("--tokenizer_type", type=str, default="bpe", choices=["bpe", "character"],
                        help="Type of tokenizer to use")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size for the tokenizer")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizer.json",
                        help="Path to save/load the tokenizer")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="decoder_only", 
                        choices=["transformer", "decoder_only"],
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
    
    # UI arguments
    parser.add_argument("--port", type=int, default=8501,
                        help="Port for Streamlit UI")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="",
                        choices=["cpu", "cuda", ""],
                        help="Device to use (empty for auto-detection)")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model",
                        help="Path to save/load the model")
    
    return parser.parse_args()


def train(args, logger):
    """Train a new model."""
    logger.info("Starting training mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}")
    try:
        text_data = load_text_file(args.data_path)
        # Split into samples (e.g., by paragraphs or documents)
        samples = [s.strip() for s in text_data.split("\n\n") if s.strip()]
        logger.info(f"Loaded {len(samples)} text samples")
        
        # Split dataset
        splits = split_dataset(
            samples, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        logger.info(f"Split dataset: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # Save splits for later use
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        save_json(
            {
                "train_size": len(splits['train']),
                "val_size": len(splits['val']),
                "test_size": len(splits['test'])
            },
            os.path.join(os.path.dirname(args.data_path), "data_splits.json")
        )
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Create and train tokenizer
    logger.info(f"Creating {args.tokenizer_type} tokenizer with vocab size {args.vocab_size}")
    try:
        if args.tokenizer_type == "bpe":
            tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        else:
            tokenizer = CharacterTokenizer(vocab_size=args.vocab_size)
        
        logger.info("Training tokenizer on data")
        tokenizer.train(splits['train'])
        
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
            "dropout": args.dropout
        }
        
        if args.model_type == "transformer":
            model = TransformerModel(model_config)
        else:
            model = DecoderOnlyTransformer(model_config)
        
        logger.info(f"Model created with {model.get_parameter_count()['total']:,} parameters")
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return
    
    # Train model
    logger.info("Starting model training")
    try:
        trained_model, history = train_model(
            model=model,
            train_texts=splits['train'],
            val_texts=splits['val'],
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            checkpoint_dir=args.checkpoint_dir,
            device=device
        )
        
        # Save training history
        save_json(
            history,
            os.path.join(args.checkpoint_dir, "training_history.json")
        )
        
        logger.info(f"Training completed. Model saved to {args.model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return


def test(args, logger):
    """Test a trained model."""
    logger.info("Starting testing mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
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
        with open(f"{args.model_path}_config.json", 'r') as f:
            import json
            config = json.load(f)
        
        # Create model instance
        if args.model_type == "transformer":
            model = TransformerModel(config)
        else:
            model = DecoderOnlyTransformer(config)
        
        # Load model weights
        model.load_state_dict(torch.load(f"{args.model_path}.pt", map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded with {model.get_parameter_count()['total']:,} parameters")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    try:
        # Check if we have data splits
        splits_path = os.path.join(os.path.dirname(args.data_path), "data_splits.json")
        if os.path.exists(splits_path):
            # Load original data and use the same splits
            text_data = load_text_file(args.data_path)
            samples = [s.strip() for s in text_data.split("\n\n") if s.strip()]
            
            splits = split_dataset(
                samples, 
                val_ratio=args.val_ratio, 
                test_ratio=args.test_ratio,
                seed=args.seed
            )
            test_texts = splits['test']
        else:
            # Just load the data as is
            text_data = load_text_file(args.data_path)
            test_texts = [s.strip() for s in text_data.split("\n\n") if s.strip()]
        
        logger.info(f"Loaded {len(test_texts)} test samples")
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return
    
    # Create tester
    logger.info("Creating model tester")
    tester = ModelTester(model, tokenizer, device)
    
    # Create prompt-completion pairs for accuracy testing
    # For simplicity, we'll use the first half of each test sample as prompt
    # and the second half as expected completion
    prompt_completion_pairs = []
    for text in test_texts[:100]:  # Limit to first 100 samples for efficiency
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
        return


def generate(args, logger):
    """Generate text using a trained model."""
    logger.info("Starting text generation mode")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
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
        with open(f"{args.model_path}_config.json", 'r') as f:
            import json
            config = json.load(f)
        
        # Create model instance
        if args.model_type == "transformer":
            model = TransformerModel(config)
        else:
            model = DecoderOnlyTransformer(config)
        
        # Load model weights
        model.load_state_dict(torch.load(f"{args.model_path}.pt", map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded with {model.get_parameter_count()['total']:,} parameters")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Check if prompt is provided
    if not args.prompt:
        logger.error("No prompt provided. Use --prompt to specify a text prompt.")
        return
    
    # Generate text
    logger.info(f"Generating text with prompt: {args.prompt}")
    try:
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
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        logger.info("Generated text:")
        print("\n" + "=" * 50)
        print(generated_text)
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return


def start_ui(args, logger):
    """Start the Streamlit UI."""
    logger.info("Starting Streamlit UI")
    
    # Check if Streamlit is installed
    try:
        import streamlit
    except ImportError:
        logger.error("Streamlit is not installed. Please install it with 'pip install streamlit'.")
        return
    
    # Start Streamlit server
    import subprocess
    import sys
    
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "app.py")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", str(args.port)
    ]
    
    logger.info(f"Starting Streamlit server on port {args.port}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        logger.info("Streamlit server stopped by user")
    except Exception as e:
        logger.error(f"Error starting Streamlit server: {str(e)}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Mini LLM Project - Mode: {args.mode}")
    
    # Execute selected mode
    if args.mode == "train":
        train(args, logger)
    elif args.mode == "test":
        test(args, logger)
    elif args.mode == "generate":
        generate(args, logger)
    elif args.mode == "ui":
        start_ui(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
