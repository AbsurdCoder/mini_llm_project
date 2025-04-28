from django.db import models
from django.utils import timezone
import os
import json

class TrainingData(models.Model):
    """Model to store uploaded training data files"""
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='training_data/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.IntegerField(default=0)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
        super().save(*args, **kwargs)

class Tokenizer(models.Model):
    """Model to store tokenizer information"""
    TOKENIZER_TYPES = (
        ('bpe', 'BPE Tokenizer'),
        ('character', 'Character Tokenizer'),
    )
    
    name = models.CharField(max_length=255)
    tokenizer_type = models.CharField(max_length=20, choices=TOKENIZER_TYPES, default='bpe')
    vocab_size = models.IntegerField(default=10000)
    file = models.FileField(upload_to='tokenizers/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.get_tokenizer_type_display()})"

class LLMModel(models.Model):
    """Model to store information about trained models"""
    MODEL_TYPES = (
        ('decoder_only', 'Decoder Only Transformer'),
        ('transformer', 'Full Transformer'),
        ('encoder_only', 'Encoder Only (BERT-like)'),
    )
    
    name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, default='decoder_only')
    tokenizer = models.ForeignKey(Tokenizer, on_delete=models.SET_NULL, null=True, blank=True)
    checkpoint_file = models.FileField(upload_to='models/', null=True, blank=True)
    config_file = models.FileField(upload_to='models/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Model architecture parameters
    d_model = models.IntegerField(default=256)
    num_heads = models.IntegerField(default=4)
    num_layers = models.IntegerField(default=4)
    d_ff = models.IntegerField(default=1024)
    max_seq_len = models.IntegerField(default=512)
    dropout = models.FloatField(default=0.1)
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"
    
    def get_config(self):
        """Return model configuration as a dictionary"""
        return {
            "model_type": self.model_type,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 10000,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout
        }
    
    def save_config(self):
        """Save model configuration to a JSON file"""
        if not self.checkpoint_file:
            return
        
        config = self.get_config()
        config_path = os.path.splitext(self.checkpoint_file.path)[0] + "_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

class TrainingSession(models.Model):
    """Model to store information about training sessions"""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE, related_name='training_sessions')
    training_data = models.ForeignKey(TrainingData, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Training parameters
    batch_size = models.IntegerField(default=16)
    learning_rate = models.FloatField(default=5e-5)
    num_epochs = models.IntegerField(default=10)
    early_stopping_patience = models.IntegerField(null=True, blank=True)
    
    # Validation parameters
    val_split = models.FloatField(default=0.1)
    
    # Results
    best_val_loss = models.FloatField(null=True, blank=True)
    best_train_loss = models.FloatField(null=True, blank=True)
    final_perplexity = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Training session for {self.model.name} ({self.status})"
    
    def start(self):
        """Mark training session as started"""
        self.status = 'running'
        self.started_at = timezone.now()
        self.save()
    
    def complete(self, val_loss=None, train_loss=None, perplexity=None):
        """Mark training session as completed"""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.best_val_loss = val_loss
        self.best_train_loss = train_loss
        self.final_perplexity = perplexity
        self.save()
    
    def fail(self):
        """Mark training session as failed"""
        self.status = 'failed'
        self.completed_at = timezone.now()
        self.save()

class TrainingProgress(models.Model):
    """Model to store training progress metrics"""
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='progress_updates')
    epoch = models.IntegerField()
    train_loss = models.FloatField()
    train_perplexity = models.FloatField()
    val_loss = models.FloatField(null=True, blank=True)
    val_perplexity = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['epoch']
    
    def __str__(self):
        return f"Epoch {self.epoch} for {self.session.model.name}"

class TextGeneration(models.Model):
    """Model to store text generation requests and results"""
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE, related_name='generations')
    prompt = models.TextField()
    generated_text = models.TextField(null=True, blank=True)
    max_length = models.IntegerField(default=100)
    temperature = models.FloatField(default=0.7)
    top_k = models.IntegerField(default=50)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Generation with {self.model.name} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"
