"""
Custom middleware for error handling in the Mini LLM UI.
"""
import logging
import traceback
import json
from django.http import JsonResponse

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware:
    """Middleware to handle exceptions and return appropriate JSON responses for API endpoints."""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        return response
    
    def process_exception(self, request, exception):
        """Process exceptions and return appropriate JSON responses for API endpoints."""
        # Only handle exceptions for API endpoints
        if request.path.startswith('/api/'):
            logger.error(f"Exception in {request.path}: {str(exception)}")
            logger.error(traceback.format_exc())
            
            # Return JSON response with error details
            return JsonResponse({
                'status': 'error',
                'message': str(exception),
                'type': exception.__class__.__name__
            }, status=500)
        
        # Let Django handle non-API exceptions
        return None


class ValidationMiddleware:
    """Middleware to validate requests for API endpoints."""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Only validate POST requests to API endpoints
        if request.method == 'POST' and request.path.startswith('/api/'):
            # Check for required parameters based on endpoint
            if request.path == '/api/upload-training-data/':
                if 'file' not in request.FILES:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'No file provided',
                        'errors': {'file': ['This field is required.']}
                    }, status=400)
            
            elif request.path == '/api/upload-tokenizer/':
                if 'name' not in request.POST or not request.POST['name']:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Tokenizer name is required',
                        'errors': {'name': ['This field is required.']}
                    }, status=400)
            
            elif request.path == '/api/upload-model/':
                if 'checkpoint_file' not in request.FILES:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'No checkpoint file provided',
                        'errors': {'checkpoint_file': ['This field is required.']}
                    }, status=400)
            
            elif request.path == '/api/generate-text/':
                if 'prompt' not in request.POST or not request.POST['prompt']:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Prompt is required',
                        'errors': {'prompt': ['This field is required.']}
                    }, status=400)
        
        response = self.get_response(request)
        return response
