"""
Test script for validating the enhanced Mini LLM functionality.

This script tests all the major components of the enhanced Mini LLM project:
1. File loading from multiple formats
2. Model creation and training
3. Tokenizer functionality
4. Model retraining
5. Text generation
6. Detailed logging

Usage:
    python test_enhanced_functionality.py
"""
import os
import sys
import logging
import tempfile
import unittest
import torch
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.file_loader import FileLoader, DataSplitter
from utils.validation import validate_file_path, validate_model_config
from tokenizers.character_tokenizer import CharacterTokenizer
from tokenizers.bpe_tokenizer import BPETokenizer
from models.transformer_model import DecoderOnlyTransformer
from models.encoder_model import EncoderOnlyModel
from training.enhanced_detailed_trainer import EnhancedTrainer
from training.model_extraction import load_model_and_tokenizer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TestEnhancedFunctionality(unittest.TestCase):
    """Test case for enhanced Mini LLM functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary test directory: {cls.test_dir}")
        
        # Create test data files
        cls.create_test_files()
        
        # Set device
        cls.device = "cpu"  # Use CPU for testing
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(cls.test_dir)
        logger.info(f"Removed temporary test directory: {cls.test_dir}")
    
    @classmethod
    def create_test_files(cls):
        """Create test data files in various formats."""
        # Create text file
        text_content = """
        This is a test file for the Mini LLM project.
        It contains multiple paragraphs of text.
        
        This is the second paragraph.
        It has multiple sentences.
        
        And here's a third paragraph.
        With some more text.
        """
        cls.text_file = os.path.join(cls.test_dir, "test.txt")
        with open(cls.text_file, "w") as f:
            f.write(text_content)
        
        # Create HTML file
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test HTML</title>
        </head>
        <body>
            <h1>Test HTML File</h1>
            <p>This is a paragraph in an HTML file.</p>
            <p>This is another paragraph with <b>bold</b> text.</p>
            <div>
                <p>This is a nested paragraph.</p>
            </div>
        </body>
        </html>
        """
        cls.html_file = os.path.join(cls.test_dir, "test.html")
        with open(cls.html_file, "w") as f:
            f.write(html_content)
        
        # Create JSON file
        json_content = """
        {
            "title": "Test JSON",
            "paragraphs": [
                "This is the first paragraph from JSON.",
                "This is the second paragraph from JSON.",
                "This is the third paragraph from JSON."
            ],
            "metadata": {
                "author": "Test Author",
                "date": "2025-04-28"
            }
        }
        """
        cls.json_file = os.path.join(cls.test_dir, "test.json")
        with open(cls.json_file, "w") as f:
            f.write(json_content)
        
        # Create checkpoint directory
        cls.checkpoint_dir = os.path.join(cls.test_dir, "checkpoints")
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
    
    def test_file_loader(self):
        """Test file loading from multiple formats."""
        logger.info("Testing file loading functionality")
        
        # Test TXT loading
        txt_content = FileLoader.load_file(self.text_file)
        self.assertIsInstance(txt_content, str)
        self.assertIn("This is a test file", txt_content)
        
        # Test HTML loading
        html_content = FileLoader.load_file(self.html_file)
        self.assertIsInstance(html_content, str)
        self.assertIn("Test HTML File", html_content)
        self.assertIn("This is a paragraph", html_content)
        
        # Test JSON loading
        json_content = FileLoader.load_file(self.json_file)
        self.assertIsInstance(json_content, str)
        self.assertIn("This is the first paragraph from JSON", json_content)
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        logger.info("Testing data splitting functionality")
        
        # Load test data
        text_data = FileLoader.load_file(self.text_file)
        
        # Test paragraph splitting
        paragraphs = DataSplitter.split_by_paragraphs(text_data)
        self.assertGreater(len(paragraphs), 1)
        
        # Test sentence splitting
        sentences = DataSplitter.split_by_sentences(text_data)
        self.assertGreater(len(sentences), len(paragraphs))
        
        # Test chunk splitting
        chunks = DataSplitter.split_by_chunks(text_data, chunk_size=50, overlap=10)
        self.assertGreater(len(chunks), 0)
        
        # Test regex splitting
        regex_chunks = DataSplitter.split_by_regex(text_data, r"\n\n+")
        self.assertEqual(len(regex_chunks), len(paragraphs))
    
    def test_tokenizers(self):
        """Test tokenizer functionality."""
        logger.info("Testing tokenizer functionality")
        
        # Load test data
        text_data = FileLoader.load_file(self.text_file)
        paragraphs = DataSplitter.split_by_paragraphs(text_data)
        
        # Test character tokenizer
        char_tokenizer = CharacterTokenizer(vocab_size=100)
        char_tokenizer.train(paragraphs)
        
        # Save and load character tokenizer
        char_tokenizer_path = os.path.join(self.test_dir, "char_tokenizer.json")
        char_tokenizer.save(char_tokenizer_path)
        loaded_char_tokenizer = CharacterTokenizer.load(char_tokenizer_path)
        
        # Test encoding and decoding
        sample_text = "This is a test."
        encoded = loaded_char_tokenizer.encode(sample_text)
        decoded = loaded_char_tokenizer.decode(encoded)
        self.assertEqual(sample_text, decoded)
        
        # Test BPE tokenizer
        bpe_tokenizer = BPETokenizer(vocab_size=100)
        bpe_tokenizer.train(paragraphs)
        
        # Save and load BPE tokenizer
        bpe_tokenizer_path = os.path.join(self.test_dir, "bpe_tokenizer.json")
        bpe_tokenizer.save(bpe_tokenizer_path)
        loaded_bpe_tokenizer = BPETokenizer.load(bpe_tokenizer_path)
        
        # Test encoding and decoding
        encoded = loaded_bpe_tokenizer.encode(sample_text)
        decoded = loaded_bpe_tokenizer.decode(encoded)
        self.assertEqual(sample_text, decoded)
    
    def test_model_creation(self):
        """Test model creation and configuration."""
        logger.info("Testing model creation")
        
        # Test decoder-only model
        decoder_config = {
            "vocab_size": 100,
            "d_model": 64,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.1,
            "verbose": True
        }
        
        # Validate config
        validate_model_config(decoder_config)
        
        # Create model
        decoder_model = DecoderOnlyTransformer(decoder_config)
        self.assertIsInstance(decoder_model, DecoderOnlyTransformer)
        
        # Test encoder-only model
        encoder_config = {
            "vocab_size": 100,
            "d_model": 64,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.1,
            "verbose": True
        }
        
        # Validate config
        validate_model_config(encoder_config)
        
        # Create model
        encoder_model = EncoderOnlyModel(encoder_config)
        self.assertIsInstance(encoder_model, EncoderOnlyModel)
    
    def test_mini_training(self):
        """Test mini training run."""
        logger.info("Testing mini training run")
        
        # Load test data
        text_data = FileLoader.load_file(self.text_file)
        paragraphs = DataSplitter.split_by_paragraphs(text_data)
        
        # Create tokenizer
        tokenizer = CharacterTokenizer(vocab_size=100)
        tokenizer.train(paragraphs)
        
        # Create model
        model_config = {
            "vocab_size": len(tokenizer.token_to_id),
            "d_model": 32,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 64,
            "max_seq_len": 64,
            "dropout": 0.1,
            "verbose": True
        }
        model = DecoderOnlyTransformer(model_config)
        
        # Prepare data
        from torch.utils.data import Dataset, DataLoader
        
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # Tokenize text
                tokens = self.tokenizer.encode(text)
                
                # Truncate or pad to max_length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                # Create input_ids and labels
                input_ids = tokens[:-1] if len(tokens) > 1 else tokens
                labels = tokens[1:] if len(tokens) > 1 else tokens
                
                # Pad to max_length - 1 (since we're using input_ids and labels)
                pad_length = self.max_length - 1 - len(input_ids)
                if pad_length > 0:
                    input_ids = input_ids + [0] * pad_length
                    labels = labels + [0] * pad_length
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * (len(tokens) - pad_length - 1) + [0] * pad_length
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
                }
        
        # Create datasets and dataloaders
        train_dataset = TextDataset(paragraphs, tokenizer, model_config["max_seq_len"])
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        
        # Create trainer
        trainer = EnhancedTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            learning_rate=1e-3,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
            early_stopping_patience=None,
            verbose=True
        )
        
        # Train for 1 epoch
        results = trainer.train(num_epochs=1)
        
        # Check results
        self.assertIn("epochs_completed", results)
        self.assertEqual(results["epochs_completed"], 1)
        self.assertIn("best_train_loss", results)
        
        # Check for saved model
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, "best_model.pt")))
    
    def test_model_loading_and_generation(self):
        """Test model loading and text generation."""
        logger.info("Testing model loading and generation")
        
        # Skip if model file doesn't exist
        model_path = os.path.join(self.checkpoint_dir, "best_model")
        if not os.path.exists(f"{model_path}.pt"):
            logger.warning("Skipping model loading test as no model file exists")
            return
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.test_dir, "char_tokenizer.json")
        if not os.path.exists(tokenizer_path):
            logger.warning("Skipping model loading test as no tokenizer file exists")
            return
        
        try:
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device=self.device
            )
            
            # Generate text
            prompt = "This is"
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Generate text
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_length=20,
                    temperature=0.7,
                    top_k=5
                )
            
            # Decode output
            output_text = tokenizer.decode(output_ids[0].tolist())
            
            # Check output
            self.assertIsInstance(output_text, str)
            self.assertTrue(output_text.startswith(prompt))
            self.assertGreater(len(output_text), len(prompt))
            
            logger.info(f"Generated text: {output_text}")
            
        except Exception as e:
            logger.error(f"Error in model loading and generation test: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"Model loading and generation test failed: {str(e)}")
    
    def test_validation(self):
        """Test validation functionality."""
        logger.info("Testing validation functionality")
        
        # Test file path validation
        valid_path = validate_file_path(self.text_file, must_exist=True)
        self.assertEqual(valid_path, self.text_file)
        
        # Test model config validation
        valid_config = {
            "vocab_size": 100,
            "d_model": 64,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.1
        }
        validated_config = validate_model_config(valid_config)
        self.assertEqual(validated_config, valid_config)
        
        # Test invalid config (should raise exception)
        invalid_config = {
            "vocab_size": 100,
            "d_model": 64,
            "num_heads": 3,  # Not divisible by d_model
            "num_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.1
        }
        with self.assertRaises(Exception):
            validate_model_config(invalid_config)


if __name__ == "__main__":
    unittest.main()
