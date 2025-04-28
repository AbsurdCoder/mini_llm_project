"""
Example script demonstrating how to use the enhanced data loading capabilities.
"""
import argparse
import logging
from pathlib import Path

from training.enhanced_data_utils import load_and_split_data, load_multiple_files
from utils.helpers import setup_logging


def main():
    """Demonstrate enhanced data loading capabilities."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Data Loading Example")
    
    # Data loading options
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data file or directory")
    parser.add_argument("--split_method", type=str, 
                        choices=["paragraph", "sentence", "chunk", "semantic", "regex"], 
                        default="paragraph",
                        help="Method to split text into samples")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Minimum length of a text sample (characters)")
    parser.add_argument("--max_length", type=int, default=10000,
                        help="Maximum length of a text sample (characters)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Data Loading Example")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    
    # Check if data_path is a directory or a file
    data_path = Path(args.data_path)
    if data_path.is_dir():
        # Load multiple files from directory
        logger.info(f"Loading multiple files from directory: {data_path}")
        file_paths = [str(f) for f in data_path.glob("*") if f.is_file()]
        logger.info(f"Found {len(file_paths)} files")
        
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
        logger.info(f"Loading single file: {data_path}")
        train_texts, val_texts, test_texts = load_and_split_data(
            file_path=args.data_path,
            split_method=args.split_method,
            min_length=args.min_length,
            max_length=args.max_length,
            val_split=args.val_split,
            test_split=args.test_split
        )
    
    # Print statistics
    total_samples = len(train_texts) + len(val_texts) + len(test_texts)
    logger.info(f"Loaded {total_samples} text samples")
    logger.info(f"Split dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    # Print sample information
    if train_texts:
        avg_length = sum(len(text) for text in train_texts) / len(train_texts)
        min_sample_len = min(len(text) for text in train_texts)
        max_sample_len = max(len(text) for text in train_texts)
        
        logger.info(f"Average sample length: {avg_length:.1f} characters")
        logger.info(f"Shortest sample: {min_sample_len} characters")
        logger.info(f"Longest sample: {max_sample_len} characters")
        
        # Print first sample
        logger.info("\nFirst sample:")
        sample = train_texts[0]
        if len(sample) > 500:
            logger.info(f"{sample[:500]}... (truncated)")
        else:
            logger.info(sample)
    
    logger.info("Data loading complete")


if __name__ == "__main__":
    main()
