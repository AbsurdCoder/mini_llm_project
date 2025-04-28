"""
Main views for the Mini LLM Django UI.
"""
import os
import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import TrainingData, Tokenizer, LLMModel, TrainingSession, TextGeneration
from .forms import (
    TrainingDataForm, TokenizerForm, ModelConfigForm, ModelUploadForm,
    TrainingSessionForm, TextGenerationForm
)

logger = logging.getLogger(__name__)

def index(request):
    """Main view for the single-page application"""
    context = {
        # Forms
        'training_data_form': TrainingDataForm(),
        'tokenizer_form': TokenizerForm(),
        'model_config_form': ModelConfigForm(),
        'model_upload_form': ModelUploadForm(),
        'training_session_form': TrainingSessionForm(),
        'text_generation_form': TextGenerationForm(),
        
        # Data for tables
        'training_data_list': TrainingData.objects.all().order_by('-uploaded_at'),
        'tokenizer_list': Tokenizer.objects.all().order_by('-created_at'),
        'model_list': LLMModel.objects.all().order_by('-created_at'),
        'training_session_list': TrainingSession.objects.all().order_by('-started_at'),
        'text_generation_list': TextGeneration.objects.all().order_by('-created_at'),
    }
    return render(request, 'llm_app/index.html', context)

@csrf_exempt
def upload_training_data(request):
    """API endpoint for uploading training data"""
    if request.method == 'POST':
        form = TrainingDataForm(request.POST, request.FILES)
        if form.is_valid():
            training_data = form.save()
            return JsonResponse({
                'status': 'success',
                'id': training_data.id,
                'name': training_data.name,
                'file_size': training_data.file.size,
                'uploaded_at': training_data.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def upload_tokenizer(request):
    """API endpoint for uploading/creating tokenizers"""
    if request.method == 'POST':
        form = TokenizerForm(request.POST, request.FILES)
        if form.is_valid():
            tokenizer = form.save()
            return JsonResponse({
                'status': 'success',
                'id': tokenizer.id,
                'name': tokenizer.name,
                'tokenizer_type': tokenizer.get_tokenizer_type_display(),
                'vocab_size': tokenizer.vocab_size
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def create_model(request):
    """API endpoint for creating model configurations"""
    if request.method == 'POST':
        form = ModelConfigForm(request.POST)
        if form.is_valid():
            model = form.save()
            return JsonResponse({
                'status': 'success',
                'id': model.id,
                'name': model.name,
                'model_type': model.get_model_type_display(),
                'tokenizer': model.tokenizer.name if model.tokenizer else None
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def upload_model(request):
    """API endpoint for uploading existing models"""
    if request.method == 'POST':
        form = ModelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.save()
            return JsonResponse({
                'status': 'success',
                'id': model.id,
                'name': model.name,
                'model_type': model.get_model_type_display(),
                'tokenizer': model.tokenizer.name if model.tokenizer else None
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

def get_training_data_list(request):
    """API endpoint to get list of training data"""
    training_data = TrainingData.objects.all().order_by('-uploaded_at')
    data = [{
        'id': item.id,
        'name': item.name,
        'file_size': item.file.size,
        'uploaded_at': item.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
    } for item in training_data]
    return JsonResponse({'status': 'success', 'data': data})

def get_tokenizer_list(request):
    """API endpoint to get list of tokenizers"""
    tokenizers = Tokenizer.objects.all().order_by('-created_at')
    data = [{
        'id': item.id,
        'name': item.name,
        'tokenizer_type': item.get_tokenizer_type_display(),
        'vocab_size': item.vocab_size,
        'created_at': item.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for item in tokenizers]
    return JsonResponse({'status': 'success', 'data': data})

def get_model_list(request):
    """API endpoint to get list of models"""
    models = LLMModel.objects.all().order_by('-created_at')
    data = [{
        'id': item.id,
        'name': item.name,
        'model_type': item.get_model_type_display(),
        'tokenizer': item.tokenizer.name if item.tokenizer else None,
        'has_checkpoint': bool(item.checkpoint_file),
        'created_at': item.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for item in models]
    return JsonResponse({'status': 'success', 'data': data})

def get_model_details(request, model_id):
    """API endpoint to get details of a specific model"""
    model = get_object_or_404(LLMModel, id=model_id)
    data = {
        'id': model.id,
        'name': model.name,
        'model_type': model.get_model_type_display(),
        'tokenizer': model.tokenizer.name if model.tokenizer else None,
        'd_model': model.d_model,
        'num_heads': model.num_heads,
        'num_layers': model.num_layers,
        'd_ff': model.d_ff,
        'max_seq_len': model.max_seq_len,
        'dropout': model.dropout,
        'has_checkpoint': bool(model.checkpoint_file),
        'created_at': model.created_at.strftime('%Y-%m-%d %H:%M:%S')
    }
    return JsonResponse({'status': 'success', 'data': data})

@csrf_exempt
def delete_training_data(request, data_id):
    """API endpoint to delete training data"""
    if request.method == 'POST':
        training_data = get_object_or_404(TrainingData, id=data_id)
        training_data.delete()
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def delete_model(request, model_id):
    """API endpoint to delete a model"""
    if request.method == 'POST':
        model = get_object_or_404(LLMModel, id=model_id)
        model.delete()
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def delete_tokenizer(request, tokenizer_id):
    """API endpoint to delete a tokenizer"""
    if request.method == 'POST':
        tokenizer = get_object_or_404(Tokenizer, id=tokenizer_id)
        tokenizer.delete()
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
