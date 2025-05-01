# Mini LLM Project - Enhanced CLI with Transfer Learning

This document provides an overview and usage instructions for the enhanced Mini LLM project, which features a comprehensive command-line interface (CLI) for all operations, including transfer learning with pre-trained Hugging Face models.

## Project Structure

```
mini_llm_project/
├── main.py                 # Unified CLI script with transfer learning support
├── tokenizers/             # Custom tokenizer implementations (BPE, Character)
│   ├── __init__.py
│   ├── base_tokenizer.py
│   ├── bpe_tokenizer.py
│   └── character_tokenizer.py
├── models/                 # Custom model implementations
│   ├── __init__.py
│   ├── base_model.py
│   ├── transformer_model.py      # Transformer (Encoder-Decoder), Decoder-Only
│   ├── encoder_model.py        # Encoder-Only (BERT-like)
│   └── enhanced_transformer_components.py # Components with detailed logging
├── training/               # Training utilities
│   ├── __init__.py
│   ├── enhanced_detailed_trainer.py # Trainer supporting custom & HF models
│   └── model_extraction.py     # Utilities for loading/saving/retraining
├── testing/                # Testing utilities
│   ├── __init__.py
│   ├── model_tester.py
│   └── test_enhanced_functionality.py # Test suite
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── file_loader.py        # Multi-format file loading
│   ├── helpers.py
│   ├── validation.py         # Input validation
│   └── hf_offline_downloader.py # Script to download HF models offline
├── data/                   # Directory for data and tokenizers
├── checkpoints/            # Directory for model checkpoints
├── hf_models/              # Recommended directory for downloaded HF models
├── test_results/           # Directory for test results
├── generations/            # Directory for generated text
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Features

*   **Unified CLI (`main.py`)**: All functionalities are accessible through a single script.
*   **Multi-Format Data Loading**: Supports `.txt`, `.html`, `.pdf`, and `.json` files.
*   **Flexible Data Splitting**: Offers various methods (`paragraphs`, `sentences`, `chunks`, `headings`, `regex`).
*   **Multiple Model Architectures**: 
    *   Custom: `transformer`, `decoder_only`, `encoder_only`.
    *   Hugging Face: Load pre-trained models like `distilbert/distilbert-base-uncased`, `openai-community/gpt2`, etc.
*   **Customizable Tokenizers**: Supports custom `bpe` and `character` tokenizers.
*   **Transfer Learning**: Fine-tune pre-trained Hugging Face models on your data.
*   **Offline Model Support**: Download Hugging Face models once and use them offline.
*   **Training & Retraining**: Train models from scratch or continue training custom/HF models from checkpoints.
*   **Detailed Logging**: Comprehensive training logs, including optional model internal operations (`--verbose`).
*   **Model Testing**: Evaluate trained models using perplexity and generate samples.
*   **Text Generation**: Generate text using trained custom or HF models.
*   **Error Handling & Validation**: Robust validation for inputs and configurations.

## Setup

1.  **Clone the repository** (or extract the project files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    # Install PyTorch separately based on your system (CPU/GPU)
    # See: https://pytorch.org/get-started/locally/
    # Example (CPU): pip install torch torchvision torchaudio
    # Example (CUDA): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Downloading Pre-trained Models for Offline Use

To use Hugging Face models offline, download them first using the provided script:

```bash
# Example: Download DistilBERT
python utils/hf_offline_downloader.py \
    --model_name distilbert/distilbert-base-uncased \
    --output_dir ./hf_models/distilbert-base-uncased

# Example: Download GPT-2
python utils/hf_offline_downloader.py \
    --model_name openai-community/gpt2 \
    --output_dir ./hf_models/gpt2
```

*   This saves the model and tokenizer files to the specified `--output_dir`.
*   You only need to do this once per model.

## Usage (`main.py`)

The `main.py` script is the primary interface. Use the `--mode` argument to select the operation.

### Common Arguments

*   `--data_path`: Path to your data file.
*   `--model_type`: `transformer`, `decoder_only`, `encoder_only` (for custom models) OR a Hugging Face model name (e.g., `distilbert/distilbert-base-uncased`).
*   `--tokenizer_type`: `bpe` or `character` (only for custom models).
*   `--tokenizer_path`: Path to save/load custom tokenizer OR path to the downloaded HF tokenizer directory.
*   `--checkpoint_dir`: Directory for model checkpoints.
*   `--model_path`: Base path for saving/loading specific model checkpoints OR path to the downloaded HF model directory.
*   `--offline_hf_dir`: Path to the directory containing the downloaded HF model/tokenizer (used with HF `--model_type`).
*   `--device`: `cpu`, `cuda`, `mps` (or empty for auto-detect).
*   `--verbose`: Enable detailed logging.
*   `--log_file`: Optional path to save logs.
*   See `python main.py --help` for all arguments.

### 1. Training a Custom Model (`--mode train`)

(Same as before, using custom `--model_type` and `--tokenizer_type`)

```bash
python main.py \
    --mode train \
    --data_path ./data/your_corpus.txt \
    --tokenizer_type bpe \
    --vocab_size 5000 \
    --tokenizer_path ./data/my_tokenizer.json \
    --model_type decoder_only \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 4 \
    --max_seq_len 256 \
    --batch_size 32 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints/custom_decoder \
    --verbose
```

### 2. Fine-tuning a Pre-trained Hugging Face Model (`--mode train`)

Use the Hugging Face model name for `--model_type` and point `--offline_hf_dir` to the downloaded model directory.

```bash
# Ensure you have downloaded the model first (see Downloading section)
python main.py \
    --mode train \
    --data_path ./data/your_finetune_data.txt \
    --model_type distilbert/distilbert-base-uncased \
    --offline_hf_dir ./hf_models/distilbert-base-uncased \
    --max_seq_len 256 \
    --batch_size 16 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --checkpoint_dir ./checkpoints/distilbert_finetuned \
    --verbose
```

*   Loads the pre-trained model and tokenizer from the offline directory.
*   Fine-tunes the model on `your_finetune_data.txt`.
*   Saves fine-tuned checkpoints (including `best_model.pt` and `config.json`) in `./checkpoints/distilbert_finetuned`.
*   Note: `--tokenizer_type`, `--vocab_size`, `--d_model`, etc., are ignored when using a Hugging Face model type, as these are loaded from the pre-trained model's configuration.

### 3. Retraining an Existing Model (`--mode retrain`)

This mode loads a previously saved checkpoint (either custom or fine-tuned HF) and continues training.

```bash
# Example: Continue fine-tuning the DistilBERT model
python main.py \
    --mode retrain \
    --data_path ./data/more_finetune_data.txt \
    --model_path ./checkpoints/distilbert_finetuned/best_model \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --checkpoint_dir ./checkpoints/distilbert_retrained \
    --verbose
```

*   Loads the model configuration and weights from the specified `--model_path` (e.g., `./checkpoints/distilbert_finetuned/best_model.pt` and associated config).
*   The tokenizer is automatically loaded based on the model config (either custom path or HF offline path).
*   Continues training on `more_finetune_data.txt`.
*   Saves new checkpoints in `./checkpoints/distilbert_retrained`.

### 4. Testing a Trained Model (`--mode test`)

Works for both custom and fine-tuned HF models.

```bash
# Test the fine-tuned DistilBERT model
python main.py \
    --mode test \
    --data_path ./data/test_data.txt \
    --model_path ./checkpoints/distilbert_finetuned/best_model \
    --test_output_dir ./test_results/distilbert \
    --batch_size 16
```

*   Loads the specified model and its corresponding tokenizer.
*   Evaluates on `test_data.txt`.
*   Saves results to `./test_results/distilbert/test_results.json`.

### 5. Generating Text (`--mode generate`)

Works for both custom and fine-tuned HF models (that support generation, e.g., GPT-2).

```bash
# Generate using a fine-tuned GPT-2 model (assuming downloaded and fine-tuned)
python main.py \
    --mode generate \
    --model_path ./checkpoints/gpt2_finetuned/best_model \
    --prompt "Once upon a time" \
    --max_length 100 \
    --temperature 0.7
```

*   Loads the specified model and tokenizer.
*   Generates text based on the prompt.
*   Prints output and saves to the `./generations` directory.

### Detailed Logging (`--verbose`)

Adding the `--verbose` flag enables detailed logs, including model internal operations for custom models and standard Hugging Face logs for pre-trained models.

## Dependencies

Key dependencies include:

*   `torch`: For model building and training.
*   `transformers`: For loading and using Hugging Face models/tokenizers.
*   `tokenizers`: (Hugging Face library) Used by HF tokenizers and custom BPE.
*   `huggingface_hub`: For downloading models.
*   `tqdm`: For progress bars.
*   `beautifulsoup4`, `requests`, `pypdf`: For loading HTML and PDF files.

See `requirements.txt` for a full list.

