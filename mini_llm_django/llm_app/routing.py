from django.urls import path
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/training/(?P<session_id>\d+)/$', consumers.TrainingConsumer.as_asgi()),
]
