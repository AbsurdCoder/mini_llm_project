"""
Example script demonstrating the fixed BPE tokenizer with proper spacing.
"""
import argparse
import logging
import os
import torch
from pathlib import Path

from ctokenizers.fixed_bpe_tokenizer import BPETokenizer
from models import DecoderOnlyTransformer
from utils.helpers import setup_logging


def main():
    """Demonstrate the fixed BPE tokenizer with proper spacing."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fixed BPE Tokenizer Demo")
    
    # Model loading options
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model",
                        help="Path to model checkpoint (without extension)")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizer.json",
                        help="Path to tokenizer file")
    
    # Generation options
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for text generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k for sampling")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Fixed BPE Tokenizer Demo")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check if tokenizer exists
    if not os.path.exists(args.tokenizer_path):
        logger.error(f"Tokenizer file not found: {args.tokenizer_path}")
        return
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    
    # First try to load with the fixed tokenizer
    try:
        tokenizer = BPETokenizer.load(args.tokenizer_path)
        logger.info("Successfully loaded tokenizer with fixed BPE implementation")
    except Exception as e:
        logger.error(f"Error loading tokenizer with fixed implementation: {str(e)}")
        logger.info("Falling back to original implementation")
        
        # Import original tokenizer as fallback
        from ctokenizers.bpe_tokenizer import BPETokenizer as OriginalBPETokenizer
        tokenizer = OriginalBPETokenizer.load(args.tokenizer_path)
    
    # Check if model exists
    config_path = f"{args.model_path}_config.json"
    model_path = f"{args.model_path}.pt"
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        logger.error(f"Model files not found: {config_path} or {model_path}")
        
        # Demonstrate tokenizer only
        logger.info("Demonstrating tokenizer only (no model)")
        
        # Encode and decode with the tokenizer
        prompt = args.prompt
        logger.info(f"Original prompt: {prompt}")
        
        # Encode
        token_ids = tokenizer.encode(prompt)
        logger.info(f"Encoded token IDs: {token_ids}")
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        logger.info(f"Decoded text: {decoded}")
        
        return
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    import json
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model = DecoderOnlyTransformer(config)
    
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Generate text
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
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
    # Decode generated text with original tokenizer
    from ctokenizers.bpe_tokenizer import BPETokenizer as OriginalBPETokenizer
    original_tokenizer = OriginalBPETokenizer.load(args.tokenizer_path)
    original_text = original_tokenizer.decode(output_ids[0].tolist())
    
    # Decode generated text with fixed tokenizer
    fixed_text = tokenizer.decode(output_ids[0].tolist())
    
    print("\nGenerated Text (Original Tokenizer):")
    print("-" * 40)
    print(original_text)
    print("-" * 40)
    
    print("\nGenerated Text (Fixed Tokenizer):")
    print("-" * 40)
    print(fixed_text)
    print("-" * 40)
    
    logger.info("Generation complete")


if __name__ == "__main__":
    main()
