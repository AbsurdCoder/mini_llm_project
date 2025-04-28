"""
Enhanced trainer with early stopping functionality.
"""
import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple

class Trainer:
    """Trainer for transformer models with early stopping."""
    
    def __init__(self, model, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-3, checkpoint_dir: str = "./checkpoints",
                 device: Optional[str] = None, early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on ('cpu', 'cuda', or None for auto-detection)
            early_stopping_patience: Number of epochs with no improvement after which training will be stopped
            early_stopping_min_delta: Minimum change in validation loss to qualify as improvement
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        
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
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.train_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            # Reshape outputs and labels for loss calculation
            # outputs: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len]
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Returns:
            Dictionary with validation metrics
        """
        # Check if validation dataloader is empty
        if self.val_dataloader is None or len(self.val_dataloader) == 0:
            self.logger.warning("Validation dataloader is empty. Skipping validation.")
            return {
                "loss": 0.0,
                "perplexity": 0.0
            }
            
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item() * labels.size(0)
                total_tokens += labels.size(0)
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None) -> str:
        """
        Save a checkpoint of the model.
        
        Args:
            is_best: Whether this is the best model so far
            filename: Optional filename to save the checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        if filename is None:
            filename = "last_model.pt"
            
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save model weights and metadata
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "patience_counter": self.patience_counter
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, save it as best_model.pt
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            
            # Save model config with the correct filename format that the UI expects
            config_path = os.path.join(self.checkpoint_dir, "best_model_config.json")
            with open(config_path, "w") as f:
                json.dump(self.model.config, f, indent=2)
                
            self.logger.info(f"Saved best model to {best_model_path}")
            
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.best_train_loss = checkpoint.get("best_train_loss", float('inf'))
        self.patience_counter = checkpoint.get("patience_counter", 0)
        
        self.logger.info(f"Loaded checkpoint from {path}")
    
    def train(self, epochs: int) -> Dict[str, Any]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting model training")
        
        try:
            for epoch in range(epochs):
                self.logger.info(f"Starting epoch {epoch+1}/{epochs}")
                
                # Train for one epoch
                train_metrics = self.train_epoch()
                
                # Log training metrics
                self.logger.info(f"Epoch {epoch+1}: "
                                f"Train Loss: {train_metrics['loss']:.4f}, "
                                f"Train Perplexity: {train_metrics['perplexity']:.2f}")
                
                # Evaluate on validation set
                val_metrics = self.evaluate()
                
                # Log validation metrics
                if self.val_dataloader is not None and len(self.val_dataloader) > 0:
                    self.logger.info(f"Epoch {epoch+1}: "
                                    f"Val Loss: {val_metrics['loss']:.4f}, "
                                    f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                
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
        
        # Save final model
        final_path = self.save_checkpoint(filename="final_model.pt")
        self.logger.info(f"Training completed. Final model saved to {final_path}")
        
        return {
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "epochs_completed": epochs if self.early_stopping_patience is None or self.patience_counter < self.early_stopping_patience else epoch + 1,
            "early_stopped": self.early_stopping_patience is not None and self.patience_counter >= self.early_stopping_patience
        }
