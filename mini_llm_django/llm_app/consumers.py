import json
from channels.generic.websocket import AsyncWebsocketConsumer

class TrainingConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time training updates"""
    
    async def connect(self):
        """Handle WebSocket connection"""
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.group_name = f'training_{self.session_id}'
        
        # Join group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Leave group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle messages received from WebSocket"""
        # Not used in this implementation, but could be used for client-to-server communication
        pass
    
    async def training_update(self, event):
        """Handle training update messages and send them to the WebSocket"""
        message = event['message']
        
        # Send message to WebSocket
        await self.send(text_data=json.dumps(message))
