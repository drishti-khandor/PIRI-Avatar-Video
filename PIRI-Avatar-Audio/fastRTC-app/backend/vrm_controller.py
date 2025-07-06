"""
Clean VRM Avatar Controller
Manages WebSocket connections and blend shape updates
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Optional
from fastapi import WebSocket
from ovr_lipsync import OVRLipsyncExtractor
from smooth_animator import SmoothAnimator, AnimationFrame

logger = logging.getLogger(__name__)

class VRMAvatarController:
    """
    Clean VRM Avatar Controller with OVRLipsync integration
    """
    
    def __init__(self):
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Core components
        self.ovr_extractor = OVRLipsyncExtractor()
        self.smooth_animator = SmoothAnimator(smoothing_factor=0.2, target_fps=60.0)
        
        # State
        self.current_emotion = "neutral"
        self.is_speaking = False
        
        # Setup animator callback
        self.smooth_animator.set_frame_callback(self._on_animation_frame)
        self.smooth_animator.start_animation()
        
        logger.info("VRM Avatar Controller initialized")
    
    async def connect(self, websocket: WebSocket):
        """Connect new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        
        # Send initial neutral state
        await self._broadcast_to_clients({
            "type": "vrm_update",
            "blend_shapes": self.smooth_animator.get_current_state(),
            "emotion": self.current_emotion,
            "timestamp": time.time()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def process_audio_chunk(self, audio_chunk, original_sample_rate: int):
        """
        Process audio chunk and update VRM blend shapes
        """
        try:
            # Extract visemes using OVRLipsync
            visemes = self.ovr_extractor.extract_visemes_from_audio(audio_chunk, original_sample_rate)
            
            if not visemes:
                logger.warning("No visemes extracted from audio")
                return
            
            # Convert to blend shapes
            blend_shapes = self.ovr_extractor.visemes_to_blend_shapes(visemes)
            
            # Update smooth animator
            self.smooth_animator.update_target(blend_shapes)
            
            # Update speaking state
            self.is_speaking = any(v.weight > 0.1 for v in visemes if v.viseme_id != 0)
            
            logger.debug(f"Processed audio: {len(visemes)} visemes -> {len(blend_shapes)} blend shapes")
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
    
    def _on_animation_frame(self, frame: AnimationFrame):
        """
        Callback for animation frames from smooth animator
        """
        try:
            # Create message for clients
            message = {
                "type": "vrm_update",
                "blend_shapes": frame.blend_shapes,
                "emotion": self.current_emotion,
                "is_speaking": self.is_speaking,
                "timestamp": frame.timestamp
            }
            
            # Send to all clients (non-blocking)
            asyncio.create_task(self._broadcast_to_clients(message))
            
        except Exception as e:
            logger.error(f"Animation frame callback error: {e}")
    
    async def _broadcast_to_clients(self, message: Dict):
        """
        Broadcast message to all connected WebSocket clients
        """
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def set_emotion(self, emotion: str):
        """Set avatar emotion"""
        if emotion in ["neutral", "happy", "sad", "surprised", "angry"]:
            self.current_emotion = emotion
            logger.info(f"Emotion set to: {emotion}")
            
            # Broadcast emotion change
            await self._broadcast_to_clients({
                "type": "emotion_change",
                "emotion": emotion,
                "timestamp": time.time()
            })
        else:
            logger.warning(f"Unknown emotion: {emotion}")
    
    async def reset_to_neutral(self):
        """Reset avatar to neutral state"""
        self.smooth_animator.reset_to_neutral()
        self.current_emotion = "neutral"
        self.is_speaking = False
        
        logger.info("Avatar reset to neutral")
    
    def get_status(self) -> Dict:
        """Get current controller status"""
        return {
            "connected_clients": len(self.active_connections),
            "current_emotion": self.current_emotion,
            "is_speaking": self.is_speaking,
            "current_blend_shapes": self.smooth_animator.get_current_state(),
            "smoothing_factor": self.smooth_animator.smoothing_factor
        }
    
    def set_smoothing_factor(self, factor: float):
        """Adjust animation smoothing"""
        self.smooth_animator.set_smoothing_factor(factor)
    
    def cleanup(self):
        """Cleanup resources"""
        self.smooth_animator.stop_animation()
        logger.info("VRM Controller cleaned up")