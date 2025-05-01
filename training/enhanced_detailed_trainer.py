"""
Enhanced trainer compatible with both custom models and Hugging Face models,
featuring detailed logging, transfer learning support, and a custom data collator.
"""
import os
import json
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm import tqdm
import math # For perplexity calculation

# Try importing Hugging Face transformers
try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_scheduler, DataCollatorWithPadding
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, MaskedLMOutput
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    PreTrainedModel, PreTrainedTokenizerBase, get_scheduler, DataCollatorWithPadding = None, None, None, None
    CausalLMOutputWithCrossAttentions, MaskedLMOutput = None, None

class TextDataset(Dataset):
    """Simple dataset for text data, returns variable length sequences."""
    def __init__(self, texts: List[str], tokenizer: Any, max_seq_len: int, is_hf_tokenizer: bool):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_hf_tokenizer = is_hf_tokenizer
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.is_hf_tokenizer:
            # Hugging Face tokenizer encoding - DO NOT PAD HERE, collator will handle it
            encoding = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors=None # Return lists, not tensors
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            # Create labels (shift for Causal LM, use input_ids for Masked LM - handled later)
            labels = input_ids[:] # Use slicing to copy
            # Masking padding tokens in labels will be handled by collator if needed, but HF usually uses -100
            # labels = [l if l != self.pad_token_id else -100 for l in labels]
        else:
            # Custom tokenizer encoding
            token_ids = self.tokenizer.encode(text)
            # Truncate
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            
            input_ids = token_ids
            attention_mask = [1] * len(token_ids)
            
            # Create labels (shift for Causal LM)
            labels = [-100] * len(token_ids) # Use -100 for ignored index
            if len(token_ids) > 1:
                labels[:len(token_ids)-1] = token_ids[1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class CustomDataCollator:
    """Pads sequences dynamically to the max length in a batch."""
    def __init__(self, tokenizer: Any, is_hf_tokenizer: bool):
        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer
        # Use tokenizer's pad_token_id if available, otherwise default (e.g., 0)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.label_pad_token_id = -100 # Standard for ignored loss calculation

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        # Determine max length in this batch
        max_len = max(len(f["input_ids"]) for f in features)

        # Pad input_ids
        input_ids_padded = []
        attention_mask_padded = []
        labels_padded = []

        for f in features:
            remainder = max_len - len(f["input_ids"])
            input_ids_padded.append(torch.tensor(f["input_ids"] + [self.pad_token_id] * remainder, dtype=torch.long))
            attention_mask_padded.append(torch.tensor(f["attention_mask"] + [0] * remainder, dtype=torch.long))
            labels_padded.append(torch.tensor(f["labels"] + [self.label_pad_token_id] * remainder, dtype=torch.long))

        batch["input_ids"] = torch.stack(input_ids_padded)
        batch["attention_mask"] = torch.stack(attention_mask_padded)
        batch["labels"] = torch.stack(labels_padded)

        return batch

class EnhancedTrainer:
    """Enhanced trainer compatible with custom and Hugging Face models."""
    
    def __init__(self, model: torch.nn.Module, tokenizer: Any,
                 train_texts: List[str], val_texts: Optional[List[str]] = None,
                 batch_size: int = 16, learning_rate: float = 5e-5, num_epochs: int = 3,
                 device: Optional[str] = None, checkpoint_dir: str = "./checkpoints",
                 model_save_path: str = "./checkpoints/best_model", # Base path for saving
                 optimizer_name: str = "adamw", scheduler_name: str = "linear",
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0, max_seq_len: int = 512,
                 verbose: bool = False, is_hf_model: bool = False,
                 start_epoch: int = 0):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train (custom nn.Module or HF PreTrainedModel)
            tokenizer: Tokenizer instance (custom or HF PreTrainedTokenizerBase)
            train_texts: List of training texts
            val_texts: Optional list of validation texts
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Total number of epochs to train for
            device: Device to train on ('cpu', 'cuda', 'mps', or None for auto-detection)
            checkpoint_dir: Directory to save checkpoints
            model_save_path: Base path for saving model files (e.g., ./checkpoints/my_model)
            optimizer_name: Optimizer type ('adam', 'adamw')
            scheduler_name: Learning rate scheduler type ('linear', 'cosine', 'constant')
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped
            early_stopping_min_delta: Minimum change in validation loss to qualify as improvement
            max_seq_len: Maximum sequence length for tokenization
            verbose: Whether to log detailed information about model internals
            is_hf_model: Flag indicating if the model is a Hugging Face model
            start_epoch: The epoch number to start training from (for resuming)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.model_save_path = model_save_path # Use this base path
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.is_hf_model = is_hf_model
        self.start_epoch = start_epoch
        
        # Set up device
        if device is None:
            if HUGGINGFACE_AVAILABLE and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    test_tensor = torch.ones(1, device='mps')
                    _ = test_tensor * 2
                    device = 'mps'
                except Exception:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(self.device)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Create datasets and dataloaders
        self.is_hf_tokenizer = HUGGINGFACE_AVAILABLE and isinstance(tokenizer, PreTrainedTokenizerBase)
        
        # Use the appropriate data collator
        if self.is_hf_tokenizer and HUGGINGFACE_AVAILABLE:
            # Use Hugging Face's default collator if available and suitable
            # Or use our custom one if HF's doesn't fit (e.g., for custom label handling)
            # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            data_collator = CustomDataCollator(tokenizer=self.tokenizer, is_hf_tokenizer=self.is_hf_tokenizer)
        else:
            data_collator = CustomDataCollator(tokenizer=self.tokenizer, is_hf_tokenizer=self.is_hf_tokenizer)
            
        self.train_dataset = TextDataset(train_texts, tokenizer, max_seq_len, self.is_hf_tokenizer)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        
        self.val_dataloader = None
        if val_texts:
            self.val_dataset = TextDataset(val_texts, tokenizer, max_seq_len, self.is_hf_tokenizer)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        # Set up optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "adamw":
            if HUGGINGFACE_AVAILABLE:
                # Use HF AdamW for potentially better weight decay handling
                from transformers import AdamW
                self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            self.logger.warning(f"Unsupported optimizer: {optimizer_name}. Defaulting to AdamW.")
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Set up learning rate scheduler
        num_training_steps = len(self.train_dataloader) * num_epochs
        if HUGGINGFACE_AVAILABLE and scheduler_name in ["linear", "cosine"]:
            self.scheduler = get_scheduler(
                name=scheduler_name,
                optimizer=self.optimizer,
                num_warmup_steps=0, # Can be configured if needed
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = None # No scheduler or simple LambdaLR if needed
            self.logger.info("Using constant learning rate (no scheduler or unsupported type).")

        # Set up loss function (only for custom models, HF models compute loss internally)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100) if not is_hf_model else None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Ensure the specific model save path directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Early stopping parameters
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        
        # Training statistics
        self.training_history = {
            "train_loss": [], "train_perplexity": [],
            "val_loss": [], "val_perplexity": [],
            "learning_rate": [], "epoch_times": []
        }
        
        self._print_model_summary()

    def _print_model_summary(self):
        """Print a summary of the model architecture and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info("=" * 50)
        self.logger.info("MODEL SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Model Type: {self.model.__class__.__name__}")
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        
        # Print model configuration if available (especially for HF models)
        if hasattr(self.model, 'config'):
            self.logger.info("\nModel Configuration:")
            try:
                config_obj = self.model.config
                config_dict = None
                if hasattr(config_obj, 'to_dict'): # Standard HF method
                    config_dict = config_obj.to_dict()
                elif isinstance(config_obj, dict): # If it's already a dict
                    config_dict = config_obj
                elif hasattr(config_obj, '__dict__'): # Fallback for custom objects
                     config_dict = vars(config_obj)
                # Add another fallback: try accessing items directly if it behaves like a dict
                elif hasattr(config_obj, 'items') and callable(config_obj.items):
                     config_dict = dict(config_obj.items())

                if config_dict:
                    for key, value in config_dict.items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        self.logger.info(f"  {key}: {value_str}")
                else:
                    # If no dict conversion worked, print the object representation
                    self.logger.info(f"  Could not convert config to dict. Config object: {config_obj}")

            except Exception as e:
                self.logger.warning(f"Could not print model config: {e}")
        
        self.logger.info(f"\nTraining Device: {self.device}")
        self.logger.info(f"Learning Rate: {self.learning_rate}")
        self.logger.info(f"Optimizer: {self.optimizer_name}")
        self.logger.info(f"Scheduler: {self.scheduler_name if self.scheduler else 'None'}")
        self.logger.info(f"Early Stopping Patience: {self.early_stopping_patience}")
        self.logger.info(f"Verbose Logging: {self.verbose}")
        self.logger.info("=" * 50)
        
    def train_epoch(self, epoch_num: int) -> Dict[str, float]:
        """
        Train for one epoch with detailed progress logging.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_dataloader)
        processed_samples = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_num+1}/{self.num_epochs} Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.verbose:
                self.logger.debug(f"Batch {batch_idx+1} shapes: input_ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}")
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.verbose:
                self.logger.debug(f"Starting forward pass for batch {batch_idx+1}")
            
            if self.is_hf_model:
                # Hugging Face models typically return a dict-like object with loss
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            else:
                # Custom model forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # Calculate loss - outputs: [batch_size, seq_len, vocab_size], labels: [batch_size, seq_len]
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            if self.verbose:
                self.logger.debug(f"Forward pass completed for batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            processed_samples += input_ids.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf') # Avoid overflow
        
        self.logger.info(f"Epoch {epoch_num+1} Training Complete. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        return {"train_loss": avg_loss, "train_perplexity": perplexity}

    def evaluate(self, epoch_num: int) -> Optional[Dict[str, float]]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with validation metrics or None if no validation set.
        """
        if not self.val_dataloader:
            return None
            
        self.model.eval()
        total_loss = 0.0
        total_batches = len(self.val_dataloader)
        
        progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch_num+1}/{self.num_epochs} Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.is_hf_model:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"})

        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        self.logger.info(f"Epoch {epoch_num+1} Validation Complete. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary containing training/validation metrics
            is_best: Flag indicating if this is the best model so far
        """
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'is_hf_model': self.is_hf_model,
            'config': self.model.config,
            'tokenizer_info': {
                'class': self.tokenizer.__class__.__name__,
                'vocab_size': getattr(self.tokenizer, 'vocab_size', None),
                'max_len': self.max_seq_len,
                'is_hf': self.is_hf_tokenizer
            }
        }
        
        # Save regular checkpoint
        # checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        # torch.save(checkpoint_state, checkpoint_path)
        # self.logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save the best model separately based on validation loss (or train loss if no validation)
        if is_best:
            best_model_path_pt = f"{self.model_save_path}.pt"
            torch.save(checkpoint_state, best_model_path_pt)
            self.logger.info(f"Best model checkpoint saved to {best_model_path_pt}")
            
            # Also save Hugging Face model/tokenizer using their standard method if applicable
            if self.is_hf_model and HUGGINGFACE_AVAILABLE:
                try:
                    self.model.save_pretrained(os.path.dirname(self.model_save_path)) # Save to directory
                    self.tokenizer.save_pretrained(os.path.dirname(self.model_save_path))
                    self.logger.info(f"Hugging Face model and tokenizer saved to {os.path.dirname(self.model_save_path)}")
                except Exception as e:
                    self.logger.error(f"Failed to save Hugging Face model/tokenizer: {e}")
            # Save custom tokenizer config if needed (assuming it has save method)
            elif not self.is_hf_tokenizer and hasattr(self.tokenizer, 'save'):
                 try:
                    tokenizer_save_path = f"{self.model_save_path}_tokenizer.json" # Or appropriate format
                    self.tokenizer.save(tokenizer_save_path)
                    self.logger.info(f"Custom tokenizer saved to {tokenizer_save_path}")
                 except Exception as e:
                    self.logger.error(f"Failed to save custom tokenizer: {e}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            self.logger.info(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            self.training_history["train_loss"].append(train_metrics["train_loss"])
            self.training_history["train_perplexity"].append(train_metrics["train_perplexity"])
            self.training_history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation
            val_metrics = self.evaluate(epoch)
            current_loss = train_metrics["train_loss"] # Default to train loss if no validation
            is_best = False
            
            if val_metrics:
                self.training_history["val_loss"].append(val_metrics["val_loss"])
                self.training_history["val_perplexity"].append(val_metrics["val_perplexity"])
                current_loss = val_metrics["val_loss"]
                
                # Check for improvement (best validation loss)
                if current_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = current_loss
                    self.patience_counter = 0
                    is_best = True
                    self.logger.info(f"Validation loss improved to {current_loss:.4f}. Saving best model.")
                else:
                    self.patience_counter += 1
                    self.logger.info(f"Validation loss did not improve significantly ({current_loss:.4f} vs best {self.best_val_loss:.4f}). Patience: {self.patience_counter}/{self.early_stopping_patience}")
            else:
                # If no validation, save based on training loss improvement
                if current_loss < self.best_train_loss - self.early_stopping_min_delta:
                    self.best_train_loss = current_loss
                    is_best = True
                    self.logger.info(f"Training loss improved to {current_loss:.4f}. Saving best model.")
                else:
                     self.logger.info(f"Training loss did not improve significantly ({current_loss:.4f} vs best {self.best_train_loss:.4f}).")
            
            # Save checkpoint (always save best model)
            self.save_checkpoint(epoch, {**train_metrics, **(val_metrics or {})}, is_best=is_best)
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.training_history["epoch_times"].append(epoch_duration)
            self.logger.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")
            
            # Early stopping check
            if self.early_stopping_patience and self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
                break
                
        end_time = time.time()
        total_duration = end_time - start_time
        self.logger.info(f"\nTraining finished in {total_duration:.2f} seconds.")
        
        # Save training history
        history_path = os.path.join(os.path.dirname(self.model_save_path), "training_history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=4)
            self.logger.info(f"Training history saved to {history_path}")
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")

        return self.training_history

