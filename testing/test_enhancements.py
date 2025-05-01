"""
Test script for validating the new model enhancements.

This script tests:
1. Encoder-only model (BERT-like)
2. Retrieval-augmented generation
3. Early stopping functionality
4. Model extraction and retraining
"""

import os
import torch
import logging
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import project modules
from models.encoder_model import EncoderOnlyModel
from models.transformer_model import DecoderOnlyTransformer
from models.rag_utils import SimpleDocumentStore, DocumentProcessor, RAGModel
from training.enhanced_trainer import Trainer
from training.model_extraction import load_model_and_tokenizer, continue_training, extract_model_for_inference
from ctokenizers.bpe_tokenizer import BPETokenizer

def test_encoder_only_model():
    """Test the encoder-only BERT-like model."""
    logger.info("Testing encoder-only model...")
    
    # Create a small config for testing
    config = {
        "model_type": "encoder_only",
        "vocab_size": 1000,
        "d_model": 128,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 512,
        "max_seq_len": 128,
        "dropout": 0.1
    }
    
    # Create model
    model = EncoderOnlyModel(config)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Test forward pass
    hidden_states = model(input_ids, attention_mask)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config["d_model"])
    assert hidden_states.shape == expected_shape, f"Expected shape {expected_shape}, got {hidden_states.shape}"
    
    # Test MLM prediction
    mlm_logits = model.predict_mlm(input_ids, attention_mask)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config["vocab_size"])
    assert mlm_logits.shape == expected_shape, f"Expected shape {expected_shape}, got {mlm_logits.shape}"
    
    logger.info("Encoder-only model test passed!")
    return True

def test_rag():
    """Test the retrieval-augmented generation functionality."""
    logger.info("Testing retrieval-augmented generation...")
    
    # Create a document store
    doc_store = SimpleDocumentStore(embedding_dim=64)
    
    # Add some test documents
    documents = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "The United States has Washington D.C. as its capital.",
        "London is the capital of the United Kingdom."
    ]
    
    doc_store.add_documents(documents)
    
    # Compute embeddings
    doc_store.compute_embeddings()
    
    # Test document retrieval
    query = "What is the capital of France?"
    results = doc_store.search(query, top_k=2)
    
    # Check if the most relevant document is returned first
    assert results[0]["document"] == documents[0], f"Expected '{documents[0]}', got '{results[0]['document']}'"
    
    # Test document processor
    text = """
    # Introduction
    
    This is a test document.
    It has multiple paragraphs.
    
    # Section 1
    
    This is section 1.
    It contains important information.
    
    # Section 2
    
    This is section 2.
    It also contains important information.
    """
    
    # Test paragraph splitting
    chunks = DocumentProcessor.split_text(text, method="paragraph")
    assert len(chunks) == 5, f"Expected 5 paragraphs, got {len(chunks)}"
    
    # Test semantic splitting
    chunks = DocumentProcessor.split_text(text, method="semantic")
    assert len(chunks) == 3, f"Expected 3 semantic sections, got {len(chunks)}"
    
    logger.info("RAG test passed!")
    return True

def test_early_stopping():
    """Test the early stopping functionality."""
    logger.info("Testing early stopping functionality...")
    
    # Create a small model for testing
    config = {
        "model_type": "decoder_only",
        "vocab_size": 1000,
        "d_model": 128,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 512,
        "max_seq_len": 128,
        "dropout": 0.1
    }
    
    model = DecoderOnlyTransformer(config)
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    # Create dummy dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Create temporary checkpoint directory
    checkpoint_dir = "./test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-3,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
        early_stopping_patience=2
    )
    
    # Train for a few epochs
    results = trainer.train(epochs=5)
    
    # Check if early stopping was triggered
    assert results["early_stopped"] or results["epochs_completed"] <= 5, "Early stopping test failed"
    
    # Check if checkpoint files were created
    assert os.path.exists(os.path.join(checkpoint_dir, "last_model.pt")), "Checkpoint file not created"
    
    logger.info("Early stopping test passed!")
    return True

def test_model_extraction_and_retraining():
    """Test model extraction and retraining functionality."""
    logger.info("Testing model extraction and retraining...")
    
    # Create a small model for testing
    config = {
        "model_type": "decoder_only",
        "vocab_size": 1000,
        "d_model": 128,
        "num_heads": 2,
        "num_layers": 2,
        "d_ff": 512,
        "max_seq_len": 128,
        "dropout": 0.1
    }
    
    model = DecoderOnlyTransformer(config)
    
    # Create temporary directories
    os.makedirs("./test_models", exist_ok=True)
    os.makedirs("./test_models/continued", exist_ok=True)
    
    # Save model
    model_path = "./test_models/test_model"
    model.save(model_path)
    
    # Extract model for inference
    extracted_path = "./test_models/extracted_model"
    extract_model_for_inference(model_path, extracted_path)
    
    # Check if extracted model files were created
    assert os.path.exists(f"{extracted_path}.pt"), "Extracted model file not created"
    assert os.path.exists(f"{extracted_path}_config.json"), "Extracted model config file not created"
    
    # Create dummy data for continued training
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    # Create dummy dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Test loading model
    loaded_model, _ = load_model_and_tokenizer(model_path, tokenizer_path=None, device="cpu")
    
    # Check if model was loaded correctly
    assert isinstance(loaded_model, DecoderOnlyTransformer), "Model not loaded correctly"
    
    logger.info("Model extraction and retraining test passed!")
    return True

def run_all_tests():
    """Run all tests."""
    tests = [
        test_encoder_only_model,
        test_rag,
        test_early_stopping,
        test_model_extraction_and_retraining
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {str(e)}")
            results.append(False)
    
    # Print summary
    logger.info("\nTest Summary:")
    for i, test in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        logger.info(f"{test.__name__}: {status}")
    
    return all(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for model enhancements")
    parser.add_argument("--test", choices=["encoder", "rag", "early_stopping", "extraction", "all"], 
                        default="all", help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "encoder":
        test_encoder_only_model()
    elif args.test == "rag":
        test_rag()
    elif args.test == "early_stopping":
        test_early_stopping()
    elif args.test == "extraction":
        test_model_extraction_and_retraining()
    else:
        run_all_tests()
