from flask import Flask
from flask_socketio import SocketIO
from websocket_handler import WebSocketHandler

def create_socket_app(app):
    # Initialize SocketIO with the Flask app
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize WebSocket handler
    ws_handler = WebSocketHandler(socketio)
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        if ws_handler.processing:
            ws_handler.stop_processing()
    
    @socketio.on('start_processing')
    def handle_start_processing():
        ws_handler.start_processing()
    
    @socketio.on('stop_processing')
    def handle_stop_processing():
        ws_handler.stop_processing()
    
    @socketio.on('process_frame')
    def handle_process_frame(frame_data):
        if ws_handler.processing:
            ws_handler.process_frame(frame_data)
    
    return socketio