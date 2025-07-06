"""
Clean VRM Avatar Controller - FIXED VISEME PROCESSING
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
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VRMAvatarController:
    """
    Clean VRM Avatar Controller with OVRLipsync integration
    FIXED: Enhanced logging and viseme processing
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
        
        # Async handling
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Setup animator callback
        self.smooth_animator.set_frame_callback(self._on_animation_frame)
        self.smooth_animator.start_animation()
        
        logger.info("VRM Avatar Controller initialized")
    
    async def connect(self, websocket: WebSocket):
        """Connect new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        
        # Store the event loop for later use
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        
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
        FIXED: Enhanced audio processing with detailed logging
        """
        try:
            logger.info(f"🎵 Processing audio chunk: {len(audio_chunk)} samples at {original_sample_rate}Hz")
            
            # Extract visemes using OVRLipsync
            visemes = self.ovr_extractor.extract_visemes_from_audio(audio_chunk, original_sample_rate)
            
            if not visemes:
                logger.warning("❌ No visemes extracted from audio")
                return
            
            logger.info(f"🎭 Extracted {len(visemes)} visemes: {[(v.viseme_id, v.weight) for v in visemes[:3]]}")
            
            # Convert to blend shapes
            blend_shapes = self.ovr_extractor.visemes_to_blend_shapes(visemes)
            
            if not blend_shapes:
                logger.warning("❌ No blend shapes generated")
                return
            
            # Log significant blend shapes
            significant_shapes = {k: v for k, v in blend_shapes.items() if v > 0.1}
            logger.info(f"📊 Generated blend shapes: {significant_shapes}")
            
            # Update smooth animator
            self.smooth_animator.update_target(blend_shapes)
            
            # Update speaking state
            self.is_speaking = any(v.weight > 0.1 for v in visemes if v.viseme_id != 0)
            
            logger.info(f"✅ Audio processed successfully. Speaking: {self.is_speaking}")
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _on_animation_frame(self, frame: AnimationFrame):
        """
        FIXED: Enhanced animation frame callback with better logging
        """
        try:
            # Check if we have any significant blend shapes
            significant_shapes = {k: v for k, v in frame.blend_shapes.items() if v > 0.01}
            
            if significant_shapes:
                logger.debug(f"🎬 Animation frame: {len(significant_shapes)} active shapes")
            
            # Create message for clients
            message = {
                "type": "vrm_update",
                "blend_shapes": frame.blend_shapes,
                "emotion": self.current_emotion,
                "is_speaking": self.is_speaking,
                "timestamp": frame.timestamp
            }
            
            # Schedule the broadcast in the main event loop
            if self.loop and not self.loop.is_closed():
                # Use call_soon_threadsafe to schedule the coroutine
                future = asyncio.run_coroutine_threadsafe(
                    self._broadcast_to_clients(message), 
                    self.loop
                )
                # Don't wait for the result to avoid blocking
                
        except Exception as e:
            logger.error(f"❌ Animation frame callback error: {e}")
    
    async def _broadcast_to_clients(self, message: Dict):
        """
        FIXED: Enhanced broadcasting with better error handling
        """
        if not self.active_connections:
            logger.debug("📡 No WebSocket connections to broadcast to")
            return
        
        # Check for significant data
        if message.get("type") == "vrm_update" and message.get("blend_shapes"):
            significant_shapes = {k: v for k, v in message["blend_shapes"].items() if v > 0.01}
            if significant_shapes:
                logger.info(f"📡 Broadcasting to {len(self.active_connections)} clients: {list(significant_shapes.keys())}")
                logger.info(f"📊 Blend shape values: {significant_shapes}")
        
        message_json = json.dumps(message)
        logger.debug(f"📤 Sending message: {message_json[:200]}...")
        
        disconnected = []
        sent_count = 0
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
                sent_count += 1
                logger.debug(f"✅ Sent to client {sent_count}")
            except Exception as e:
                logger.error(f"❌ Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
        
        if sent_count > 0:
            logger.debug(f"✅ Broadcast successful to {sent_count}/{len(self.active_connections)} clients")
        else:
            logger.warning("❌ Failed to broadcast to any clients")
    
    async def set_emotion(self, emotion: str):
        """Set avatar emotion"""
        if emotion in ["neutral", "happy", "sad", "surprised", "angry"]:
            self.current_emotion = emotion
            logger.info(f"😊 Emotion set to: {emotion}")
            
            # Broadcast emotion change
            await self._broadcast_to_clients({
                "type": "emotion_change",
                "emotion": emotion,
                "timestamp": time.time()
            })
        else:
            logger.warning(f"❌ Unknown emotion: {emotion}")
    
    async def reset_to_neutral(self):
        """Reset avatar to neutral state"""
        self.smooth_animator.reset_to_neutral()
        self.current_emotion = "neutral"
        self.is_speaking = False
        
        logger.info("🔄 Avatar reset to neutral")
    
    def get_status(self) -> Dict:
        """Get current controller status"""
        current_shapes = self.smooth_animator.get_current_state()
        active_shapes = {k: v for k, v in current_shapes.items() if v > 0.01}
        
        return {
            "connected_clients": len(self.active_connections),
            "current_emotion": self.current_emotion,
            "is_speaking": self.is_speaking,
            "active_blend_shapes": active_shapes,
            "total_blend_shapes": len(current_shapes),
            "smoothing_factor": self.smooth_animator.smoothing_factor,
            "animator_running": self.smooth_animator.is_animating
        }
    
    def set_smoothing_factor(self, factor: float):
        """Adjust animation smoothing"""
        self.smooth_animator.set_smoothing_factor(factor)
        logger.info(f"🎛️ Smoothing factor set to: {factor}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.smooth_animator.stop_animation()
        self.executor.shutdown(wait=False)
        logger.info("🧹 VRM Controller cleaned up")