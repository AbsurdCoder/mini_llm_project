"""
Enhanced trainer with detailed logging of progress and model internals.
"""
import os
import json
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

class EnhancedTrainer:
    """Enhanced trainer for transformer models with detailed logging and early stopping."""
    
    def __init__(self, model, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-3, checkpoint_dir: str = "./checkpoints",
                 device: Optional[str] = None, early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0, verbose: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on ('cpu', 'cuda', 'mps', or None for auto-detection)
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped
            early_stopping_min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Whether to log detailed information about model internals
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        # Set up device
        if device is None:
            # Try to use MPS (Mac GPU) if available
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Test MPS with a small operation
                    test_tensor = torch.ones(1, device='mps')
                    _ = test_tensor * 2
                    device = 'mps'
                except:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
        self.device = device
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        
        # Training statistics
        self.training_history = {
            "train_loss": [],
            "train_perplexity": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rate": [],
            "epoch_times": []
        }
        
        # Print model summary
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
        
        # Print model configuration if available
        if hasattr(self.model, 'config'):
            self.logger.info("\nModel Configuration:")
            for key, value in self.model.config.items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info(f"\nTraining Device: {self.device}")
        self.logger.info(f"Learning Rate: {self.learning_rate}")
        self.logger.info(f"Early Stopping Patience: {self.early_stopping_patience}")
        self.logger.info(f"Verbose Logging: {self.verbose}")
        self.logger.info("=" * 50)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with detailed progress logging.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        batch_count = len(self.train_dataloader)
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Log batch progress
            if batch_idx % max(1, batch_count // 10) == 0:  # Log every 10% of batches
                self.logger.info(f"Training batch {batch_idx+1}/{batch_count} ({(batch_idx+1)/batch_count*100:.1f}%)")
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Log batch shapes if verbose
            if self.verbose:
                self.logger.debug(f"Batch {batch_idx+1} shapes: input_ids={input_ids.shape}, "
                                 f"attention_mask={attention_mask.shape}, labels={labels.shape}")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Log that we're starting forward pass
            if self.verbose:
                self.logger.debug(f"Starting forward pass for batch {batch_idx+1}")
                
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            # Reshape outputs and labels for loss calculation
            # outputs: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len]
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss = self.criterion(outputs, labels)
            
            # Log loss value
            if self.verbose:
                self.logger.debug(f"Batch {batch_idx+1} loss: {loss.item():.4f}")
            
            # Backward pass
            if self.verbose:
                self.logger.debug(f"Starting backward pass for batch {batch_idx+1}")
                
            loss.backward()
            
            # Log gradient statistics if verbose
            if self.verbose:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                self.logger.debug(f"Batch {batch_idx+1} gradient norm: {grad_norm:.4f}")
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Epoch completed: {batch_count} batches, {total_tokens} tokens processed")
        self.logger.info(f"Average training loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_batches": batch_count,
            "total_tokens": total_tokens
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data with detailed logging.
        
        Returns:
            Dictionary with validation metrics
        """
        # Check if validation dataloader is empty
        if self.val_dataloader is None or len(self.val_dataloader) == 0:
            self.logger.warning("Validation dataloader is empty. Skipping validation.")
            return {
                "loss": 0.0,
                "perplexity": 0.0,
                "total_batches": 0,
                "total_tokens": 0
            }
            
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batch_count = len(self.val_dataloader)
        
        self.logger.info(f"Starting validation on {batch_count} batches")
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.val_dataloader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Log batch progress
                if self.verbose and batch_idx % max(1, batch_count // 5) == 0:  # Log every 20% of batches
                    self.logger.debug(f"Validation batch {batch_idx+1}/{batch_count} ({(batch_idx+1)/batch_count*100:.1f}%)")
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.verbose:
                    self.logger.debug(f"Starting validation forward pass for batch {batch_idx+1}")
                    
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item() * labels.size(0)
                total_tokens += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Validation completed: {batch_count} batches, {total_tokens} tokens processed")
        self.logger.info(f"Average validation loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_batches": batch_count,
            "total_tokens": total_tokens
        }
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None) -> str:
        """
        Save a checkpoint of the model with detailed logging.
        
        Args:
            is_best: Whether this is the best model so far
            filename: Optional filename to save the checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        if filename is None:
            filename = "last_model.pt"
            
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Save model weights and metadata
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "patience_counter": self.patience_counter,
            "training_history": self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, save it as best_model.pt
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            
            # Save model config with the correct filename format
            config_path = os.path.join(self.checkpoint_dir, "best_model_config.json")
            
            # Ensure model.config exists
            model_config = getattr(self.model, 'config', {})
            if not model_config:
                # Create a basic config if none exists
                model_config = {
                    "model_type": self.model.__class__.__name__,
                    "vocab_size": getattr(self.model, 'vocab_size', 0),
                    "d_model": getattr(self.model, 'd_model', 0),
                    "num_heads": getattr(self.model, 'num_heads', 0),
                    "num_layers": getattr(self.model, 'num_layers', 0),
                    "d_ff": getattr(self.model, 'd_ff', 0),
                    "max_seq_len": getattr(self.model, 'max_seq_len', 0),
                    "dropout": getattr(self.model, 'dropout', 0.1),
                    "verbose": self.verbose
                }
            
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
                
            self.logger.info(f"Saved best model to {best_model_path}")
            self.logger.info(f"Saved model config to {config_path}")
            
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint with detailed logging.
        
        Args:
            path: Path to the checkpoint
        """
        self.logger.info(f"Loading checkpoint from {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            self.best_train_loss = checkpoint.get("best_train_loss", float('inf'))
            self.patience_counter = checkpoint.get("patience_counter", 0)
            
            # Load training history if available
            if "training_history" in checkpoint:
                self.training_history = checkpoint["training_history"]
                self.logger.info(f"Loaded training history with {len(self.training_history['train_loss'])} epochs")
            
            self.logger.info(f"Successfully loaded checkpoint from {path}")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Best training loss: {self.best_train_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def train(self, epochs: int) -> Dict[str, Any]:
        """
        Train the model for a specified number of epochs with detailed logging.
        
        Args:
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("=" * 50)
        self.logger.info(f"STARTING MODEL TRAINING FOR {epochs} EPOCHS")
        self.logger.info("=" * 50)
        self.logger.info(f"Training device: {self.device}")
        self.logger.info(f"Training data: {len(self.train_dataloader)} batches")
        if self.val_dataloader:
            self.logger.info(f"Validation data: {len(self.val_dataloader)} batches")
        self.logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info("=" * 50)
        
        # Track overall training time
        training_start_time = time.time()
        
        try:
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                self.logger.info("=" * 50)
                self.logger.info(f"EPOCH {epoch+1}/{epochs}")
                self.logger.info("=" * 50)
                
                # Train for one epoch
                self.logger.info(f"Starting training for epoch {epoch+1}")
                train_metrics = self.train_epoch()
                
                # Log training metrics
                self.logger.info(f"Epoch {epoch+1} training completed:")
                self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
                self.logger.info(f"  Train Perplexity: {train_metrics['perplexity']:.2f}")
                self.logger.info(f"  Batches processed: {train_metrics['total_batches']}")
                self.logger.info(f"  Tokens processed: {train_metrics['total_tokens']:,}")
                
                # Evaluate on validation set
                if self.val_dataloader is not None and len(self.val_dataloader) > 0:
                    self.logger.info(f"Starting validation for epoch {epoch+1}")
                    val_metrics = self.evaluate()
                    
                    # Log validation metrics
                    self.logger.info(f"Epoch {epoch+1} validation completed:")
                    self.logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                    self.logger.info(f"  Val Perplexity: {val_metrics['perplexity']:.2f}")
                    self.logger.info(f"  Batches processed: {val_metrics['total_batches']}")
                    self.logger.info(f"  Tokens processed: {val_metrics['total_tokens']:,}")
                else:
                    val_metrics = {"loss": float('inf'), "perplexity": float('inf')}
                    self.logger.info("No validation data available, skipping validation")
                
                # Calculate epoch time
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                
                self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
                
                # Update training history
                self.training_history["train_loss"].append(train_metrics["loss"])
                self.training_history["train_perplexity"].append(train_metrics["perplexity"])
                self.training_history["val_loss"].append(val_metrics["loss"])
                self.training_history["val_perplexity"].append(val_metrics["perplexity"])
                self.training_history["learning_rate"].append(self.learning_rate)
                self.training_history["epoch_times"].append(epoch_time)
                
                # Check if this is the best model
                if self.val_dataloader is not None and len(self.val_dataloader) > 0:
                    # Use validation loss if available
                    current_loss = val_metrics["loss"]
                    is_best = current_loss < self.best_val_loss - self.early_stopping_min_delta
                    
                    if is_best:
                        self.best_val_loss = current_loss
                        self.patience_counter = 0
                        self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                    else:
                        self.patience_counter += 1
                        self.logger.info(f"No improvement in validation loss. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                else:
                    # Use training loss if no validation data
                    current_loss = train_metrics["loss"]
                    is_best = current_loss < self.best_train_loss - self.early_stopping_min_delta
                    
                    if is_best:
                        self.best_train_loss = current_loss
                        self.patience_counter = 0
                        self.logger.info(f"New best training loss: {self.best_train_loss:.4f}")
                    else:
                        self.patience_counter += 1
                        self.logger.info(f"No improvement in training loss. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                # Save checkpoint
                self.save_checkpoint(is_best=is_best)
                
                # Early stopping check
                if self.early_stopping_patience is not None and self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # Save the current model as last_model.pt
            self.save_checkpoint(is_best=False)
        
        # Calculate total training time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        # Save final model
        final_path = self.save_checkpoint(filename="final_model.pt")
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # Log final training summary
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total training time: {total_training_time:.2f} seconds")
        self.logger.info(f"Epochs completed: {len(self.training_history['train_loss'])}")
        
        if self.val_dataloader is not None and len(self.val_dataloader) > 0:
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Best validation perplexity: {torch.exp(torch.tensor(self.best_val_loss)).item():.2f}")
        else:
            self.logger.info(f"Best training loss: {self.best_train_loss:.4f}")
            self.logger.info(f"Best training perplexity: {torch.exp(torch.tensor(self.best_train_loss)).item():.2f}")
            
        self.logger.info(f"Early stopping triggered: {self.early_stopping_patience is not None and self.patience_counter >= self.early_stopping_patience}")
        self.logger.info(f"Final model saved to {final_path}")
        self.logger.info(f"Training history saved to {history_path}")
        self.logger.info("=" * 50)
        
        return {
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "epochs_completed": len(self.training_history['train_loss']),
            "early_stopped": self.early_stopping_patience is not None and self.patience_counter >= self.early_stopping_patience,
            "total_training_time": total_training_time,
            "training_history": self.training_history
        }
