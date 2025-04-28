# Mini LLM Project Enhancements Documentation

This document provides detailed information about the new features and enhancements added to the Mini LLM project.

## Table of Contents
1. [Encoder-Only Model (BERT-like)](#encoder-only-model-bert-like)
2. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
3. [Early Stopping Functionality](#early-stopping-functionality)
4. [Model Extraction and Retraining](#model-extraction-and-retraining)
5. [Usage Examples](#usage-examples)

## Encoder-Only Model (BERT-like)

We've added a new encoder-only transformer model similar to BERT, which is designed for tasks like masked language modeling (MLM).

### Key Features
- Bidirectional attention (unlike decoder-only models which use causal attention)
- Masked Language Modeling (MLM) prediction head
- Compatible with the existing tokenizers and training pipeline
- Follows the same configuration pattern as other models

### Usage

```python
from models.encoder_model import EncoderOnlyModel

# Create a new encoder-only model
config = {
    "model_type": "encoder_only",
    "vocab_size": 10000,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 1024,
    "max_seq_len": 512,
    "dropout": 0.1
}

model = EncoderOnlyModel(config)

# Forward pass for getting hidden states
hidden_states = model(input_ids, attention_mask)

# Forward pass for MLM prediction
mlm_logits = model.predict_mlm(input_ids, attention_mask)
```

### Training for Masked Language Modeling

To train the encoder-only model for masked language modeling, you'll need to:

1. Randomly mask some percentage of input tokens (typically 15%)
2. Use the original tokens as labels for the masked positions
3. Train the model to predict the original tokens at masked positions

## Retrieval-Augmented Generation (RAG)

We've implemented a lightweight Retrieval-Augmented Generation (RAG) system that enhances text generation by retrieving relevant information from a document store.

### Key Features
- Simple document store with vector embeddings
- Flexible document processing with multiple splitting methods
- Integration with the existing model architecture
- Minimal dependencies (uses numpy for vector operations)

### Usage

```python
from models.rag_utils import SimpleDocumentStore, DocumentProcessor, RAGModel

# Create a document store
doc_store = SimpleDocumentStore(embedding_dim=256)

# Process and add documents
text = "Your long document text here..."
chunks = DocumentProcessor.split_text(
    text, 
    method="semantic",  # Options: paragraph, sentence, chunk, semantic, regex
    min_length=50,
    max_length=500
)

# Add documents to the store
doc_store.add_documents(chunks)

# Compute embeddings
doc_store.compute_embeddings()

# Save the document store for later use
doc_store.save("./data/document_store.pkl")

# Create a RAG model with your language model
from training.model_extraction import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer("./checkpoints/best_model")

rag_model = RAGModel(doc_store, model)

# Generate text with RAG
response = rag_model.generate(
    query="What is the capital of France?",
    top_k=3,  # Number of documents to retrieve
    max_length=100,
    temperature=0.7
)
```

### Document Splitting Methods

The `DocumentProcessor` supports multiple text splitting methods:

- **paragraph**: Split by double newlines (default)
- **sentence**: Split by sentence boundaries
- **chunk**: Split into fixed-size chunks with configurable overlap
- **semantic**: Split by headings and section boundaries
- **regex**: Split using a custom regex pattern

## Early Stopping Functionality

We've enhanced the training process with early stopping functionality to prevent overfitting and save training time.

### Key Features
- Patience-based validation loss tracking
- Automatic saving of the best model
- Support for both validation and training loss monitoring
- Configurable minimum improvement threshold

### Usage

```python
from training.enhanced_trainer import Trainer

# Create a trainer with early stopping
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=1e-3,
    checkpoint_dir="./checkpoints",
    device="cuda",  # or "mps" for Mac GPU, "cpu" for CPU
    early_stopping_patience=5,  # Stop after 5 epochs with no improvement
    early_stopping_min_delta=0.01  # Minimum improvement to count as progress
)

# Train with early stopping
results = trainer.train(epochs=30)  # Will stop early if triggered

# Check if early stopping was triggered
if results["early_stopped"]:
    print(f"Training stopped early after {results['epochs_completed']} epochs")
```

### Command Line Usage

You can also use early stopping from the command line:

```bash
python main.py --mode train --data_path ./data/train.txt --early_stopping_patience 5
```

## Model Extraction and Retraining

We've added functionality to extract trained models and continue training from checkpoints, making it easy to reuse and fine-tune your models.

### Key Features
- Load models and tokenizers from checkpoints
- Continue training from saved checkpoints
- Extract models for inference
- Support for all model types (encoder-only, transformer, decoder-only)

### Usage: Loading a Model

```python
from training.model_extraction import load_model_and_tokenizer

# Load a model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    model_path="./checkpoints/best_model",
    tokenizer_path="./data/tokenizer.json",  # Optional, will try to infer if not provided
    device="cuda"  # or "mps" for Mac GPU, "cpu" for CPU
)
```

### Usage: Continuing Training

```python
from training.model_extraction import continue_training

# Continue training from a checkpoint
results = continue_training(
    model_path="./checkpoints/best_model",
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=1e-4,  # Lower learning rate for fine-tuning
    epochs=5,
    checkpoint_dir="./checkpoints/continued",
    early_stopping_patience=3
)
```

### Usage: Extracting for Inference

```python
from training.model_extraction import extract_model_for_inference

# Extract a model for inference
extracted_path = extract_model_for_inference(
    model_path="./checkpoints/best_model",
    output_path="./models/inference_model",
    device="cpu"  # Extract on CPU for maximum compatibility
)
```

## Usage Examples

### Training an Encoder-Only Model

```python
# In main.py or a custom script

from models.encoder_model import EncoderOnlyModel
from training.enhanced_trainer import Trainer
from tokenizers.bpe_tokenizer import BPETokenizer
from training.data_utils import load_and_split_data, create_dataloaders

# Load and prepare data
train_texts, val_texts, _ = load_and_split_data(
    file_path="./data/train.txt",
    split_method="paragraph",
    val_split=0.1,
    test_split=0.0
)

# Create or load tokenizer
tokenizer = BPETokenizer(vocab_size=10000)
tokenizer.train(train_texts)
tokenizer.save("./data/tokenizer.json")

# Create dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    train_texts=train_texts,
    val_texts=val_texts,
    tokenizer=tokenizer,
    batch_size=16,
    max_length=512
)

# Create encoder-only model
config = {
    "model_type": "encoder_only",
    "vocab_size": tokenizer.vocab_size,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 1024,
    "max_seq_len": 512,
    "dropout": 0.1
}

model = EncoderOnlyModel(config)

# Create trainer with early stopping
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=1e-3,
    checkpoint_dir="./checkpoints/encoder_model",
    early_stopping_patience=5
)

# Train model
results = trainer.train(epochs=20)
```

### Using RAG with a Trained Model

```python
# In a custom script

from models.rag_utils import SimpleDocumentStore, DocumentProcessor, RAGModel
from training.model_extraction import load_model_and_tokenizer

# Load your trained model
model, tokenizer = load_model_and_tokenizer("./checkpoints/best_model")

# Create a document store
doc_store = SimpleDocumentStore(embedding_dim=256)

# Load and process documents
with open("./data/knowledge_base.txt", "r") as f:
    text = f.read()

# Split text into chunks
chunks = DocumentProcessor.split_text(
    text, 
    method="semantic",
    min_length=50,
    max_length=500
)

# Add documents to the store
doc_store.add_documents(chunks)

# Compute embeddings
doc_store.compute_embeddings()

# Create a RAG model
rag_model = RAGModel(doc_store, model)

# Generate text with RAG
response = rag_model.generate(
    query="What is machine learning?",
    top_k=3,
    max_length=200,
    temperature=0.7
)

print(response)
```

### Fine-tuning a Pre-trained Model

```python
# In a custom script

from training.model_extraction import continue_training
from training.data_utils import load_and_split_data, create_dataloaders
from tokenizers.bpe_tokenizer import BPETokenizer

# Load new training data
train_texts, val_texts, _ = load_and_split_data(
    file_path="./data/new_data.txt",
    split_method="paragraph",
    val_split=0.1,
    test_split=0.0
)

# Load existing tokenizer
tokenizer = BPETokenizer.load("./data/tokenizer.json")

# Create dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    train_texts=train_texts,
    val_texts=val_texts,
    tokenizer=tokenizer,
    batch_size=16,
    max_length=512
)

# Continue training from a checkpoint
results = continue_training(
    model_path="./checkpoints/best_model",
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=5e-5,  # Lower learning rate for fine-tuning
    epochs=10,
    checkpoint_dir="./checkpoints/fine_tuned",
    early_stopping_patience=3
)
```
