# Mini LLM Project - Enhanced CLI

This document provides an overview and usage instructions for the enhanced Mini LLM project, which features a comprehensive command-line interface (CLI) for all operations.

## Project Structure

```
mini_llm_project/
├── main.py                 # Unified CLI script
├── tokenizers/             # Tokenizer implementations (BPE, Character)
│   ├── __init__.py
│   ├── base_tokenizer.py
│   ├── bpe_tokenizer.py
│   └── character_tokenizer.py
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── base_model.py
│   ├── transformer_model.py      # Transformer (Encoder-Decoder), Decoder-Only
│   ├── encoder_model.py        # Encoder-Only (BERT-like)
│   └── enhanced_transformer_components.py # Components with detailed logging
├── training/               # Training utilities
│   ├── __init__.py
│   ├── enhanced_detailed_trainer.py # Trainer with detailed logging
│   └── model_extraction.py     # Utilities for loading/saving/retraining
├── testing/                # Testing utilities
│   ├── __init__.py
│   ├── model_tester.py
│   └── test_enhanced_functionality.py # Test suite
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── file_loader.py        # Multi-format file loading
│   ├── helpers.py
│   └── validation.py         # Input validation
├── data/                   # Directory for data and tokenizers
├── checkpoints/            # Directory for model checkpoints
├── test_results/           # Directory for test results
├── generations/            # Directory for generated text
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Features

*   **Unified CLI (`main.py`)**: All functionalities are accessible through a single script with command-line arguments.
*   **Multi-Format Data Loading**: Supports loading training/testing data from `.txt`, `.html`, `.pdf`, and `.json` files.
*   **Flexible Data Splitting**: Offers various methods to split text data into samples (`paragraphs`, `sentences`, `chunks`, `headings`, `regex`).
*   **Multiple Model Architectures**: 
    *   `transformer`: Standard Encoder-Decoder Transformer.
    *   `decoder_only`: GPT-style Decoder-Only Transformer.
    *   `encoder_only`: BERT-style Encoder-Only Transformer.
*   **Customizable Tokenizers**: Supports `bpe` (Byte Pair Encoding) and `character` tokenizers.
*   **Training & Retraining**: Train models from scratch or continue training from existing checkpoints.
*   **Detailed Logging**: Provides comprehensive logging of the training process, including step/epoch progress and optionally detailed model internal operations (using `--verbose`).
*   **Model Testing**: Evaluate trained models using perplexity and generate qualitative samples.
*   **Text Generation**: Generate text using trained models with various sampling parameters (temperature, top-k, top-p).
*   **Error Handling & Validation**: Includes robust validation for input arguments and configurations.

## Setup

1.  **Clone the repository** (or extract the project files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    # Install PyTorch separately based on your system (CPU/GPU)
    # See: https://pytorch.org/get-started/locally/
    # Example (CPU): pip install torch torchvision torchaudio
    # Example (CUDA): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install necessary libraries for PDF/HTML parsing
    pip install beautifulsoup4 requests pypdf
    ```

## Usage (`main.py`)

The `main.py` script is the primary interface. Use the `--mode` argument to select the operation.

### Common Arguments

*   `--data_path`: Path to your data file.
*   `--tokenizer_type`: `bpe` or `character`.
*   `--tokenizer_path`: Path to save/load tokenizer.
*   `--model_type`: `transformer`, `decoder_only`, or `encoder_only`.
*   `--checkpoint_dir`: Directory for model checkpoints.
*   `--model_path`: Base path for saving/loading specific model checkpoints (e.g., `./checkpoints/my_model`).
*   `--device`: `cpu`, `cuda`, `mps` (or empty for auto-detect).
*   `--verbose`: Enable detailed logging, including model internal operations.
*   `--log_file`: Optional path to save logs to a file.
*   See `python main.py --help` for all available arguments.

### 1. Training a New Model (`--mode train`)

This mode trains a new model from scratch using the specified data and configuration.

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
    --checkpoint_dir ./checkpoints \
    --verbose  # Optional: for detailed logging
```

*   This will load data from `your_corpus.txt`, train a BPE tokenizer, save it to `my_tokenizer.json`, create a Decoder-Only model, train it for 5 epochs, and save checkpoints (including the best model based on validation loss) in the `./checkpoints` directory.
*   The model configuration will be saved as `model_config.json` in the checkpoint directory.
*   The best model weights will be saved as `best_model.pt`.

### 2. Retraining an Existing Model (`--mode retrain`)

This mode loads an existing model checkpoint and continues training on new (or the same) data.

```bash
python main.py \
    --mode retrain \
    --data_path ./data/more_data.txt \
    --model_path ./checkpoints/best_model \
    --tokenizer_path ./data/my_tokenizer.json \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --checkpoint_dir ./checkpoints/continued \
    --verbose
```

*   This loads the model configuration and weights from `./checkpoints/best_model*` and the tokenizer from `./data/my_tokenizer.json`.
*   It then continues training on `more_data.txt` for 3 more epochs.
*   New checkpoints will be saved in a subdirectory (e.g., `./checkpoints/continued`).
*   **Note**: Ensure `--tokenizer_path` points to the tokenizer compatible with the loaded model.

### 3. Testing a Trained Model (`--mode test`)

This mode evaluates a trained model on a test dataset, calculating perplexity and generating sample outputs.

```bash
python main.py \
    --mode test \
    --data_path ./data/test_data.txt \
    --model_path ./checkpoints/best_model \
    --tokenizer_path ./data/my_tokenizer.json \
    --test_output_dir ./test_results \
    --batch_size 16
```

*   Loads the model and tokenizer.
*   Loads test data from `test_data.txt`.
*   Calculates perplexity on the test set.
*   Generates a few sample continuations.
*   Saves results (perplexity, samples) to `./test_results/test_results.json`.

### 4. Generating Text (`--mode generate`)

This mode uses a trained model to generate text based on a given prompt.

```bash
python main.py \
    --mode generate \
    --model_path ./checkpoints/best_model \
    --tokenizer_path ./data/my_tokenizer.json \
    --prompt "The future of AI is" \
    --max_length 150 \
    --temperature 0.8 \
    --top_k 40
```

*   Loads the model and tokenizer.
*   Generates text starting with the `--prompt`.
*   Uses sampling parameters like `--max_length`, `--temperature`, `--top_k`, `--top_p`.
*   Prints the generated text to the console.
*   Saves the generation to a timestamped file in the `./generations` directory.
*   If `--prompt` is omitted, it will ask for input interactively.

### Detailed Logging (`--verbose`)

Adding the `--verbose` flag to any mode will enable detailed logging, including:

*   **Training**: Batch-level loss, learning rate changes, gradient norms.
*   **Model Internals**: Messages indicating when specific components are activated (e.g., "Entering Encoder Layer", "Applying Attention Mechanism", "Using Positional Encoding").

This is useful for debugging and understanding the model's behavior during training and inference.

## Dependencies

Key dependencies include:

*   `torch`: For model building and training.
*   `tokenizers`: (Hugging Face library) Used by the BPE tokenizer.
*   `tqdm`: For progress bars.
*   `beautifulsoup4`, `requests`, `pypdf`: For loading HTML and PDF files.

See `requirements.txt` for a full list.

