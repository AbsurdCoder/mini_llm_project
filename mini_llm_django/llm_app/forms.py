from django import forms
from .models import TrainingData, Tokenizer, LLMModel, TrainingSession, TextGeneration

class TrainingDataUploadForm(forms.ModelForm):
    """Form for uploading training data files"""
    class Meta:
        model = TrainingData
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter a name for this dataset'}),
            'file': forms.FileInput(attrs={'class': 'form-control'})
        }

class TokenizerForm(forms.ModelForm):
    """Form for creating or uploading tokenizers"""
    class Meta:
        model = Tokenizer
        fields = ['name', 'tokenizer_type', 'vocab_size', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter a name for this tokenizer'}),
            'tokenizer_type': forms.Select(attrs={'class': 'form-control'}),
            'vocab_size': forms.NumberInput(attrs={'class': 'form-control', 'min': '100', 'max': '50000'}),
            'file': forms.FileInput(attrs={'class': 'form-control'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].required = False
        self.fields['file'].help_text = 'Optional. Upload a pre-trained tokenizer file.'

class ModelUploadForm(forms.ModelForm):
    """Form for uploading pre-trained models"""
    config_file = forms.FileField(required=False, widget=forms.FileInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = LLMModel
        fields = ['name', 'model_type', 'tokenizer', 'checkpoint_file', 'config_file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter a name for this model'}),
            'model_type': forms.Select(attrs={'class': 'form-control'}),
            'tokenizer': forms.Select(attrs={'class': 'form-control'}),
            'checkpoint_file': forms.FileInput(attrs={'class': 'form-control'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['tokenizer'].queryset = Tokenizer.objects.all()
        self.fields['tokenizer'].empty_label = "Select a tokenizer"
        self.fields['config_file'].help_text = 'Optional. Upload the model config JSON file.'

class ModelConfigForm(forms.ModelForm):
    """Form for configuring a new model"""
    class Meta:
        model = LLMModel
        fields = ['name', 'model_type', 'tokenizer', 'd_model', 'num_heads', 'num_layers', 'd_ff', 'max_seq_len', 'dropout']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter a name for this model'}),
            'model_type': forms.Select(attrs={'class': 'form-control'}),
            'tokenizer': forms.Select(attrs={'class': 'form-control'}),
            'd_model': forms.NumberInput(attrs={'class': 'form-control', 'min': '64', 'max': '1024'}),
            'num_heads': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '16'}),
            'num_layers': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '12'}),
            'd_ff': forms.NumberInput(attrs={'class': 'form-control', 'min': '128', 'max': '4096'}),
            'max_seq_len': forms.NumberInput(attrs={'class': 'form-control', 'min': '64', 'max': '2048'}),
            'dropout': forms.NumberInput(attrs={'class': 'form-control', 'min': '0.0', 'max': '0.5', 'step': '0.1'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['tokenizer'].queryset = Tokenizer.objects.all()
        self.fields['tokenizer'].empty_label = "Select a tokenizer"

class TrainingSessionForm(forms.ModelForm):
    """Form for configuring a training session"""
    training_data = forms.ModelChoiceField(
        queryset=TrainingData.objects.all(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label="Select training data"
    )
    
    class Meta:
        model = TrainingSession
        fields = ['training_data', 'batch_size', 'learning_rate', 'num_epochs', 'early_stopping_patience', 'val_split']
        widgets = {
            'batch_size': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '128'}),
            'learning_rate': forms.NumberInput(attrs={'class': 'form-control', 'min': '0.00001', 'max': '0.1', 'step': '0.00001'}),
            'num_epochs': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '100'}),
            'early_stopping_patience': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '20'}),
            'val_split': forms.NumberInput(attrs={'class': 'form-control', 'min': '0.05', 'max': '0.3', 'step': '0.05'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['early_stopping_patience'].required = False
        self.fields['early_stopping_patience'].help_text = 'Optional. Number of epochs with no improvement after which training will be stopped.'

class TextGenerationForm(forms.ModelForm):
    """Form for text generation"""
    class Meta:
        model = TextGeneration
        fields = ['model', 'prompt', 'max_length', 'temperature', 'top_k']
        widgets = {
            'model': forms.Select(attrs={'class': 'form-control'}),
            'prompt': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter your prompt here...'}),
            'max_length': forms.NumberInput(attrs={'class': 'form-control', 'min': '10', 'max': '1000'}),
            'temperature': forms.NumberInput(attrs={'class': 'form-control', 'min': '0.1', 'max': '2.0', 'step': '0.1'}),
            'top_k': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '100'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model'].queryset = LLMModel.objects.filter(checkpoint_file__isnull=False)
        self.fields['model'].empty_label = "Select a trained model"
