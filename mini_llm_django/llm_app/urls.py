from django.urls import path
from . import views
from . import training_views

urlpatterns = [
    path('', views.index, name='index'),
    
    # API endpoints for data management
    path('api/upload-training-data/', views.upload_training_data, name='upload_training_data'),
    path('api/upload-tokenizer/', views.upload_tokenizer, name='upload_tokenizer'),
    path('api/upload-model/', views.upload_model, name='upload_model'),
    path('api/create-model/', views.create_model, name='create_model'),
    path('api/training-data/', views.get_training_data_list, name='get_training_data_list'),
    path('api/tokenizers/', views.get_tokenizer_list, name='get_tokenizer_list'),
    path('api/models/', views.get_model_list, name='get_model_list'),
    path('api/models/<int:model_id>/', views.get_model_details, name='get_model_details'),
    path('api/training-data/<int:data_id>/delete/', views.delete_training_data, name='delete_training_data'),
    path('api/models/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('api/tokenizers/<int:tokenizer_id>/delete/', views.delete_tokenizer, name='delete_tokenizer'),
    
    # API endpoints for training and generation
    path('api/models/<int:model_id>/train/', training_views.start_training, name='start_training'),
    path('api/models/<int:model_id>/retrain/', training_views.start_retraining, name='start_retraining'),
    path('api/training-sessions/<int:session_id>/progress/', training_views.get_training_progress, name='get_training_progress'),
    path('api/generate-text/', training_views.generate_text, name='generate_text'),
]
