"""
Example script demonstrating how to continue training an existing model.
"""
import argparse
import logging
import os
import torch
import json
from pathlib import Path

from tokenizers import BPETokenizer, CharacterTokenizer
from models import TransformerModel, DecoderOnlyTransformer
from training.enhanced_data_utils import load_and_split_data, create_dataloaders
from training.trainer import Trainer
from utils.helpers import setup_logging


def main():
    """Demonstrate continued training of an existing model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Continued Training Example")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to existing model checkpoint (without extension)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to new training data file")
    
    # Optional arguments
    parser.add_argument("--model_type", type=str, choices=["transformer", "decoder_only"], default="decoder_only",
                        help="Type of model to use")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizer.json",
                        help="Path to tokenizer file")
    parser.add_argument("--tokenizer_type", type=str, choices=["bpe", "character"], default="bpe",
                        help="Type of tokenizer to use")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/fine_tuned",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--learning_rate", type=float, default=0.00005,
                        help="Learning rate (recommended to use lower rate for fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--split_method", type=str, 
                        choices=["paragraph", "sentence", "chunk", "semantic", "regex"], 
                        default="paragraph",
                        help="Method to split text into samples")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Continued Training Example")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
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
    train_texts, val_texts, _ = load_and_split_data(
        file_path=args.data_path,
        split_method=args.split_method,
        val_split=0.1,
        test_split=0.0  # No test set needed for fine-tuning
    )
    
    logger.info(f"Loaded {len(train_texts) + len(val_texts)} new text samples")
    logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=config.get("max_seq_len", 512)
    )
    
    # Create trainer with loaded model
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,  # Lower learning rate for fine-tuning
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Continue training
    logger.info("Starting continued training")
    trainer.train(epochs=args.epochs)
    
    logger.info("Continued training complete")
    logger.info(f"Fine-tuned model saved to {args.checkpoint_dir}/best_model.pt")
    
    # Print example usage for generating text with the fine-tuned model
    print("\nTo generate text with your fine-tuned model, use:")
    print(f"python main.py --mode generate --model_path {args.checkpoint_dir}/best_model --prompt \"Your prompt here\"")


if __name__ == "__main__":
    main()
