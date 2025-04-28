"""
Training utilities for language models.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from tqdm import tqdm
import logging
import json
from pathlib import Path


class Trainer:
    """Trainer class for language models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        max_grad_norm: float = 1.0,
        log_interval: int = 10
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            max_grad_norm: Maximum gradient norm for gradient clipping
            log_interval: Interval for logging training progress
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index (0)
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_perplexity = 0.0
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            # Reshape outputs: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            logits = outputs.view(-1, outputs.size(-1))
            # Reshape labels: [batch_size, seq_len] -> [batch_size * seq_len]
            labels = labels.view(-1)
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            perplexity = torch.exp(loss).item()
            epoch_perplexity += perplexity
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity:.2f}"
            })
            
            # Log progress
            if (step + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.epoch + 1} | Step {step + 1}/{len(self.train_dataloader)} | "
                    f"Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}"
                )
            
            self.global_step += 1
        
        pbar.close()
        
        # Calculate epoch metrics
        epoch_loss /= len(self.train_dataloader)
        epoch_perplexity /= len(self.train_dataloader)
        epoch_time = time.time() - start_time
        
        return {
            "loss": epoch_loss,
            "perplexity": epoch_perplexity,
            "time": epoch_time
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_perplexity = 0.0
        
        # Check if validation dataloader is empty
        if len(self.val_dataloader) == 0:
            self.logger.warning("Validation dataloader is empty. Skipping validation.")
            return {
                "loss": 0.0,
                "perplexity": 0.0
            }
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                logits = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                
                loss = self.criterion(logits, labels)
                
                # Update metrics
                val_loss += loss.item()
                val_perplexity += torch.exp(loss).item()
        
        # Calculate validation metrics
        val_loss /= len(self.val_dataloader)
        val_perplexity /= len(self.val_dataloader)
        
        return {
            "loss": val_loss,
            "perplexity": val_perplexity
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save a model checkpoint.
        
        Args:
            metrics: Dictionary with training metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.epoch + 1}.pt")
        
        # Save checkpoint
        torch.save({
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "best_val_loss": self.best_val_loss
        }, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model")
            best_model_file = f"{best_model_path}.pt"
            
            # Save model weights
            torch.save({
                "epoch": self.epoch + 1,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
            }, best_model_file)
            
            # Save model config with the correct filename format that the UI expects
            config_path = f"{best_model_path}_config.json"
            with open(config_path, "w") as f:
                json.dump(self.model.config, f, indent=2)
            
            self.logger.info(f"Saved best model to {best_model_file} and config to {config_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_perplexity": [],
            "val_loss": [],
            "val_perplexity": []
        }
        
        for _ in range(num_epochs):
            self.logger.info(f"Starting epoch {self.epoch + 1}/{num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Evaluate on validation set
            val_metrics = self.evaluate()
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_perplexity"].append(train_metrics["perplexity"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_perplexity"].append(val_metrics["perplexity"])
            
            # Log metrics
            self.logger.info(
                f"Epoch {self.epoch + 1}/{num_epochs} completed in {train_metrics['time']:.2f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Perplexity: {train_metrics['perplexity']:.2f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Perplexity: {val_metrics['perplexity']:.2f}"
            )
            
            # Check if this is the best model
            # If there's no validation data, use training loss instead
            if len(self.val_dataloader) == 0:
                is_best = train_metrics["loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_metrics["loss"]
                    self.logger.info(f"New best training loss: {self.best_val_loss:.4f}")
            else:
                is_best = val_metrics["loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["loss"]
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.save_checkpoint({**train_metrics, **val_metrics}, is_best)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            self.epoch += 1
        
        self.logger.info("Training completed!")
        return history


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type.lower() == "adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "linear",
    num_warmup_steps: int = 0,
    num_training_steps: int = 0,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('linear', 'cosine', 'constant', 'step')
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type.lower() == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
    elif scheduler_type.lower() == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
    elif scheduler_type.lower() == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps
        )
    elif scheduler_type.lower() == "step":
        step_size = kwargs.get("step_size", 1)
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_model(
    model: nn.Module,
    train_texts: List[str],
    val_texts: List[str],
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    optimizer_type: str = "adamw",
    scheduler_type: str = "linear",
    weight_decay: float = 0.01,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a language model.
    
    Args:
        model: Model to train
        train_texts: List of training text samples
        val_texts: List of validation text samples
        tokenizer: Tokenizer instance for encoding texts
        batch_size: Batch size
        max_length: Maximum sequence length
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate
        optimizer_type: Type of optimizer
        scheduler_type: Type of scheduler
        weight_decay: Weight decay factor
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    from .data_utils import create_dataloaders
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train model
    history = trainer.train(num_epochs=num_epochs)
    
    return model, history
