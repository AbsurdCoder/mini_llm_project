"""
Utility script to download Hugging Face models and tokenizers for offline use.

Example Usage:
    python utils/hf_offline_downloader.py --model_name distilbert/distilbert-base-uncased --output_dir ./hf_models/distilbert-base-uncased
    python utils/hf_offline_downloader.py --model_name openai-community/gpt2 --output_dir ./hf_models/gpt2
"""

import argparse
import os
import logging

# Try importing huggingface_hub
try:
    from huggingface_hub import snapshot_download
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    snapshot_download = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Fixed syntax error here
logger = logging.getLogger(__name__)

def download_model_for_offline(
    model_name: str,
    output_dir: str,
    cache_dir: str = None,
    resume_download: bool = True,
    token: str = None
):
    """
    Downloads a Hugging Face model and tokenizer to a specified directory for offline use.

    Args:
        model_name: The name of the model on the Hugging Face Hub (e.g., "distilbert/distilbert-base-uncased").
        output_dir: The local directory where the model files should be saved.
        cache_dir: Optional path to a directory to use as cache. Defaults to Hugging Face default cache.
        resume_download: Whether to resume downloads if interrupted.
        token: Optional Hugging Face Hub token for private models.
    """
    if not HUGGINGFACE_HUB_AVAILABLE:
        logger.error("huggingface_hub library not found. Please install it: pip install huggingface_hub")
        return

    logger.info(f"Starting download for model: {model_name}")
    logger.info(f"Target directory: {output_dir}")

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Use snapshot_download to get all files
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False, # Important: Copy files instead of symlinking
            cache_dir=cache_dir,
            resume_download=resume_download,
            token=token,
            # Consider adding ignore_patterns if needed to exclude large files like .safetensors if only .bin is needed
            # ignore_patterns=["*.safetensors"]
        )

        logger.info(f"Successfully downloaded model 	{model_name}	 to 	{output_dir}	")

    except Exception as e:
        logger.error(f"Failed to download model 	{model_name}	: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models for offline use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model on Hugging Face Hub (e.g., distilbert/distilbert-base-uncased)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local directory to save the model files")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional cache directory for downloads")
    parser.add_argument("--token", type=str, default=None,
                        help="Optional Hugging Face Hub token for private models")
    parser.add_argument("--no_resume", action="store_false", dest="resume",
                        help="Disable resuming interrupted downloads")

    args = parser.parse_args()

    download_model_for_offline(
        model_name=args.model_name,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        resume_download=args.resume,
        token=args.token
    )

if __name__ == "__main__":
    main()

