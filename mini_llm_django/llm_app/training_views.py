from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import LLMModel, TrainingSession
from .forms import TrainingSessionForm
from .training_manager import TrainingManager

@csrf_exempt
def start_training(request, model_id):
    """View for starting a new training session"""
    if request.method == 'POST':
        model = get_object_or_404(LLMModel, id=model_id)
        form = TrainingSessionForm(request.POST)
        
        if form.is_valid():
            # Create training session
            session = form.save(commit=False)
            session.model = model
            session.save()
            
            # Start training in background
            TrainingManager.start_training_session(session.id)
            
            return JsonResponse({
                'status': 'success',
                'session_id': session.id,
                'message': 'Training started successfully'
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def start_retraining(request, model_id):
    """View for starting a retraining session on an existing model"""
    if request.method == 'POST':
        model = get_object_or_404(LLMModel, id=model_id)
        
        # Check if model has a checkpoint
        if not model.checkpoint_file:
            return JsonResponse({
                'status': 'error', 
                'message': 'Cannot retrain: model has no checkpoint file'
            }, status=400)
        
        form = TrainingSessionForm(request.POST)
        
        if form.is_valid():
            # Create training session
            session = form.save(commit=False)
            session.model = model
            session.save()
            
            # Start training in background (retraining uses the same process)
            TrainingManager.start_training_session(session.id)
            
            return JsonResponse({
                'status': 'success',
                'session_id': session.id,
                'message': 'Retraining started successfully'
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

def get_training_progress(request, session_id):
    """API to get training progress for a session"""
    session = get_object_or_404(TrainingSession, id=session_id)
    progress_updates = session.progress_updates.all().order_by('epoch')
    
    data = {
        'session': {
            'id': session.id,
            'status': session.get_status_display(),
            'started_at': session.started_at.strftime('%Y-%m-%d %H:%M:%S') if session.started_at else None,
            'completed_at': session.completed_at.strftime('%Y-%m-%d %H:%M:%S') if session.completed_at else None,
            'best_val_loss': session.best_val_loss,
            'best_train_loss': session.best_train_loss,
            'final_perplexity': session.final_perplexity,
        },
        'progress': [{
            'epoch': update.epoch,
            'train_loss': update.train_loss,
            'train_perplexity': update.train_perplexity,
            'val_loss': update.val_loss,
            'val_perplexity': update.val_perplexity,
            'timestamp': update.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for update in progress_updates]
    }
    
    return JsonResponse(data)

@csrf_exempt
def generate_text(request):
    """View for generating text with a trained model"""
    from .models import TextGeneration
    from .forms import TextGenerationForm
    from .training_manager import TextGenerationManager
    
    if request.method == 'POST':
        form = TextGenerationForm(request.POST)
        if form.is_valid():
            # Save generation request
            generation = form.save()
            
            # Generate text
            generated_text = TextGenerationManager.generate_text(generation.id)
            
            return JsonResponse({
                'status': 'success',
                'id': generation.id,
                'generated_text': generated_text
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
