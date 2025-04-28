# Mini LLM Project

A modular project for building, training, and using a small foundational language model from scratch.

## Features

- **Custom Tokenizers**: Both BPE (Byte Pair Encoding) and character-level tokenizers
- **Transformer Architecture**: Small but complete transformer implementation with attention mechanisms
- **Training Pipeline**: Full training loop with optimization and checkpointing
- **Testing Module**: Evaluation metrics including perplexity and accuracy
- **Streamlit UI**: Interactive interface for text generation and model inspection
- **Modular Design**: Clean separation of components for easy customization

## Project Structure

- `tokenizers/`: Custom tokenizer implementations
  - `base_tokenizer.py`: Abstract base class for tokenizers
  - `character_tokenizer.py`: Character-level tokenizer
  - `bpe_tokenizer.py`: Byte Pair Encoding tokenizer
- `models/`: Transformer model architecture
  - `base_model.py`: Abstract base class for models
  - `transformer_components.py`: Attention, feed-forward, and positional encoding
  - `transformer_model.py`: Complete transformer implementations
- `training/`: Training pipeline and utilities
  - `data_utils.py`: Dataset and dataloader utilities
  - `trainer.py`: Training loop and optimization
- `testing/`: Evaluation and accuracy testing
  - `model_tester.py`: Metrics calculation and evaluation
- `ui/`: Streamlit interface
  - `app.py`: Interactive UI for model interaction
- `utils/`: Helper functions
  - `helpers.py`: Utility functions for the project
- `data/`: Directory for training and testing datasets
- `main.py`: Command-line interface for all functionality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mini_llm_project.git
cd mini_llm_project
```

2. Install dependencies:
```bash
pip install torch numpy tqdm streamlit nltk rouge
```

## Usage

### Training a Model

To train a model from scratch:

```bash
python main.py --mode train --data_path ./data/your_corpus.txt --tokenizer_type bpe --model_type decoder_only
```

Key parameters:
- `--tokenizer_type`: Choose between `bpe` or `character`
- `--model_type`: Choose between `transformer` or `decoder_only`
- `--d_model`: Model dimension (default: 256)
- `--num_layers`: Number of transformer layers (default: 4)
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 10)

### Generating Text

To generate text with a trained model:

```bash
python main.py --mode generate --prompt "Your text prompt here" --model_path ./checkpoints/best_model --tokenizer_path ./data/tokenizer.json
```

Key parameters:
- `--max_length`: Maximum generation length (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_k`: Top-k sampling parameter (default: 50)

### Testing a Model

To evaluate a trained model:

```bash
python main.py --mode test --data_path ./data/test_corpus.txt --model_path ./checkpoints/best_model --tokenizer_path ./data/tokenizer.json
```

### Using the Streamlit UI

To launch the interactive UI:

```bash
python main.py --mode ui --port 8501
```

Then open your browser at `http://localhost:8501`

## Training Your Own Dataset

1. Prepare your text corpus in a file (e.g., `data/corpus.txt`).
2. Train a model:
```bash
python main.py --mode train --data_path ./data/corpus.txt --tokenizer_type bpe --model_type decoder_only --num_epochs 20 --batch_size 32
```
3. The trained model and tokenizer will be saved to the specified paths.
4. Use the model for generation or evaluation.

## Customization

The modular design allows for easy customization:

- **Tokenizers**: Extend `BaseTokenizer` to create new tokenization methods
- **Models**: Extend `BaseModel` to implement different architectures
- **Training**: Modify `Trainer` to implement custom training loops
- **Evaluation**: Add new metrics to `ModelTester`

## Example Workflow

1. **Prepare Data**: Create a text corpus for training
2. **Train Tokenizer and Model**: Run training with desired parameters
3. **Evaluate Model**: Test the model's performance
4. **Generate Text**: Use the model to generate new content
5. **Interact via UI**: Explore the model through the Streamlit interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.
