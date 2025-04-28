# Mini LLM Project: End-to-End Usage Guide

This guide provides comprehensive instructions for using the Mini LLM project, from environment setup to training and testing your model.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Project Structure](#project-structure)
3. [Training Process](#training-process)
4. [Testing and Evaluation](#testing-and-evaluation)
5. [Using the Streamlit UI](#using-the-streamlit-ui)
6. [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone or download the project**
   Extract the ZIP file to your desired location.

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv mini_llm_env
   # On Windows
   mini_llm_env\Scripts\activate
   # On macOS/Linux
   source mini_llm_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd mini_llm_project
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized into several modules:

- **tokenizers/**: Custom tokenizers (BPE and character-based)
- **models/**: Transformer model architecture
- **training/**: Training pipeline and utilities
- **testing/**: Evaluation and metrics
- **ui/**: Streamlit user interface
- **utils/**: Helper functions
- **data/**: Directory for storing datasets and tokenizer files
- **checkpoints/**: Directory for saving model checkpoints

## Training Process

### Step 1: Prepare Your Dataset

1. **Format your data**
   - Create a text file with your training data
   - Each document/sample should be separated by double newlines
   - Place the file in the `data/` directory

   Example data format:
   ```
   This is the first document in the training set.
   It can span multiple lines.

   This is the second document.
   It is separated from the first by a blank line.
   ```

2. **Data considerations**
   - For best results, use at least a few hundred KB of text
   - Ensure the text is clean and formatted consistently
   - Consider preprocessing (removing special characters, normalizing whitespace)

### Step 2: Train the Model

1. **Basic training command**
   ```bash
   python main.py --mode train --data_path ./data/your_corpus.txt
   ```

2. **Advanced training options**
   ```bash
   python main.py --mode train \
     --data_path ./data/your_corpus.txt \
     --model_type decoder_only \
     --tokenizer_type bpe \
     --vocab_size 10000 \
     --d_model 256 \
     --n_layers 4 \
     --n_heads 4 \
     --d_ff 1024 \
     --max_seq_len 512 \
     --dropout 0.1 \
     --learning_rate 0.0001 \
     --batch_size 16 \
     --epochs 10 \
     --val_split 0.1 \
     --test_split 0.1 \
     --checkpoint_dir ./checkpoints
   ```

3. **Training parameters explained**
   - `--model_type`: Model architecture (`transformer` or `decoder_only`)
   - `--tokenizer_type`: Tokenizer type (`bpe` or `character`)
   - `--vocab_size`: Size of the vocabulary for the tokenizer
   - `--d_model`: Dimension of the model (embedding size)
   - `--n_layers`: Number of transformer layers
   - `--n_heads`: Number of attention heads
   - `--d_ff`: Dimension of the feed-forward network
   - `--max_seq_len`: Maximum sequence length
   - `--dropout`: Dropout rate for regularization
   - `--learning_rate`: Learning rate for optimization
   - `--batch_size`: Batch size for training
   - `--epochs`: Number of training epochs
   - `--val_split`: Fraction of data to use for validation
   - `--test_split`: Fraction of data to use for testing
   - `--checkpoint_dir`: Directory to save model checkpoints

### Step 3: Monitor Training Progress

During training, you'll see output like this:
```
INFO - Starting training mode
INFO - Using device: cpu
INFO - Loading data from ./data/your_corpus.txt
INFO - Loaded 1000 text samples
INFO - Split dataset: 800 train, 100 val, 100 test
INFO - Creating bpe tokenizer with vocab size 10000
INFO - Training tokenizer on data
INFO - Tokenizer saved to ./data/tokenizer.json
INFO - Creating decoder_only model
INFO - Model created with 3,199,567 parameters
INFO - Starting model training
INFO - Starting epoch 1/10
Epoch 1: 100%|██████████| 50/50 [01:23<00:00, 1.67s/it, loss=4.2560, ppl=70.53]
INFO - Validation loss: 4.1234, perplexity: 61.82
INFO - New best validation loss: 4.1234
INFO - Saved checkpoint to ./checkpoints/best_model.pt
```

Key metrics to watch:
- **Training loss**: Should decrease over time
- **Perplexity**: Should decrease over time (lower is better)
- **Validation loss**: Should decrease without diverging too much from training loss

### Step 4: Examine Training Results

After training completes, you'll find:
- Trained model saved at `./checkpoints/best_model.pt`
- Model configuration saved at `./checkpoints/best_model_config.json`
- Tokenizer saved at `./data/tokenizer.json`

## Testing and Evaluation

### Step 1: Evaluate the Model

1. **Run evaluation on test set**
   ```bash
   python main.py --mode test \
     --model_path ./checkpoints/best_model \
     --tokenizer_path ./data/tokenizer.json \
     --data_path ./data/your_corpus.txt \
     --test_split 0.1
   ```

2. **Evaluation metrics**
   The test mode will output metrics such as:
   - **Test loss**: Overall loss on the test set
   - **Perplexity**: Measure of how well the model predicts the test samples
   - **Accuracy**: Token prediction accuracy (if applicable)

### Step 2: Generate Text with the Model

1. **Basic text generation**
   ```bash
   python main.py --mode generate \
     --model_path ./checkpoints/best_model \
     --tokenizer_path ./data/tokenizer.json \
     --prompt "Your prompt text here" \
     --max_length 100
   ```

2. **Advanced generation options**
   ```bash
   python main.py --mode generate \
     --model_path ./checkpoints/best_model \
     --tokenizer_path ./data/tokenizer.json \
     --prompt "Your prompt text here" \
     --max_length 200 \
     --temperature 0.8 \
     --top_k 40
   ```

3. **Generation parameters explained**
   - `--prompt`: Input text to start generation
   - `--max_length`: Maximum length of generated text
   - `--temperature`: Controls randomness (higher = more random)
   - `--top_k`: Number of highest probability tokens to consider

## Using the Streamlit UI

The Streamlit UI provides a user-friendly interface for interacting with your trained model.

### Step 1: Launch the UI

```bash
cd mini_llm_project
streamlit run ui/rewritten_app.py
```

### Step 2: Load Your Model

1. In the sidebar, configure:
   - **Model Type**: Select the type of model you trained (`decoder_only` or `transformer`)
   - **Tokenizer Type**: Select the tokenizer you used (`bpe` or `character`)
   - **Model Path**: Enter the path to your model (default: `./checkpoints/best_model`)
   - **Tokenizer Path**: Enter the path to your tokenizer (default: `./data/tokenizer.json`)

2. Click the "Load Model" button

### Step 3: Generate Text

1. Navigate to the "Text Generation" tab
2. Enter your prompt in the text area
3. Adjust generation parameters if needed:
   - **Maximum Length**: Length of text to generate
   - **Temperature**: Controls randomness
   - **Top-K**: Number of tokens to consider
4. Click "Generate Text"

### Step 4: View Model Information

The "Model Info" tab displays:
- Model architecture details
- Parameter count
- Device information
- Tokenizer information

### Step 5: Batch Processing (Optional)

The "Batch Processing" tab allows you to:
1. Upload a text file with multiple prompts (one per line)
2. Process all prompts at once
3. Download the results as a JSON file

## Troubleshooting

### Common Issues and Solutions

1. **"Error loading model: Config file not found"**
   - Ensure the model path is correct
   - Check that `best_model_config.json` exists in the specified directory

2. **"Error loading tokenizer"**
   - Verify the tokenizer path is correct
   - Ensure the tokenizer type matches what was used during training

3. **"CUDA out of memory"**
   - Reduce batch size
   - Reduce model size (d_model, n_layers, n_heads)
   - Use CPU instead of GPU

4. **"The size of tensor a must match the size of tensor b"**
   - This indicates a dimension mismatch in the model
   - Ensure consistent dimensions throughout the model configuration

5. **"float division by zero"**
   - This can occur when validation dataset is empty
   - Ensure your dataset is large enough to be split properly

### Getting Help

If you encounter issues not covered in this guide:
1. Check the error messages for specific information
2. Review the code documentation in each module
3. Examine the logs for detailed error traces

## Advanced Usage

### Fine-tuning an Existing Model

```bash
python main.py --mode train \
  --data_path ./data/new_corpus.txt \
  --model_path ./checkpoints/existing_model \
  --tokenizer_path ./data/existing_tokenizer.json \
  --epochs 5 \
  --learning_rate 0.00005
```

### Exporting Model for Production

The trained model can be used in production environments:

```python
from tokenizers import BPETokenizer
from models import DecoderOnlyTransformer
import torch
import json

# Load tokenizer
tokenizer = BPETokenizer.load("./data/tokenizer.json")

# Load model config
with open("./checkpoints/best_model_config.json", 'r') as f:
    config = json.load(f)

# Create model
model = DecoderOnlyTransformer(config)

# Load weights
checkpoint = torch.load("./checkpoints/best_model.pt", map_location="cpu")
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

# Set to evaluation mode
model.eval()

# Generate text
def generate(prompt, max_length=100, temperature=0.7, top_k=50):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    return tokenizer.decode(output_ids[0].tolist())
```

This guide should help you navigate the complete process of training and using your Mini LLM model. If you have any specific questions or encounter issues, please refer to the detailed documentation in each module.
