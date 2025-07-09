"""
Enhanced Viseme Controller with synchronization and smooth transitions
"""
import asyncio
import time
import logging
from typing import Dict, Set, Optional, List
from fastapi import WebSocket

from app.models.avatar import (
    VisemeData, AvatarState, AvatarUpdate, EmotionType, BlendShape
)
from app.config.settings import settings


logger = logging.getLogger(__name__)


class VisemeAnimationController:
    """
    Manages viseme animations, WebSocket connections, and synchronization
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.current_state = AvatarState()
        self.animation_queue: asyncio.Queue[VisemeData] = asyncio.Queue()
        self.is_animating = False
        self._animation_task: Optional[asyncio.Task] = None
        
        # Viseme to blend shape mappings
        self.viseme_to_blend_shape_mapping = self._initialize_viseme_mappings()
        
    def _initialize_viseme_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize blend shape mappings for each viseme"""
        return {
            "sil": {},  # Neutral/silent
            "AA": {"aa": 1.0},
            "AE": {"aa": 0.8, "ee": 0.2},
            "AH": {"aa": 0.6},
            "AO": {"oh": 0.9},
            "AW": {"aa": 0.4, "oh": 0.6},
            "AY": {"aa": 0.7, "ih": 0.3},
            "CH": {"ch": 0.8, "ih": 0.2},
            "DD": {"dd": 0.7, "nn": 0.3},
            "EH": {"ee": 0.8, "aa": 0.2},
            "ER": {"aa": 0.4, "oh": 0.3},
            "EY": {"ee": 0.9, "ih": 0.1},
            "FF": {"ff": 1.0},
            "HH": {"aa": 0.2},
            "IH": {"ih": 1.0},
            "IY": {"ih": 0.8, "ee": 0.2},
            "JH": {"ch": 0.7, "ee": 0.3},
            "KK": {"kk": 0.8, "aa": 0.1},
            "LL": {"dd": 0.4, "aa": 0.3},
            "MM": {"mm": 1.0},
            "NN": {"nn": 1.0},
            "NG": {"nn": 0.7, "kk": 0.3},
            "OW": {"oh": 1.0},
            "OY": {"oh": 0.7, "ih": 0.3},
            "PP": {"pp": 1.0},
            "RR": {"rr": 0.8, "aa": 0.2},
            "SS": {"ss": 1.0},
            "SH": {"ch": 1.0},
            "TH": {"th": 0.8, "ss": 0.2},
            "TT": {"dd": 0.8, "ss": 0.2},
            "UH": {"oh": 0.5, "aa": 0.3},
            "UW": {"oh": 0.8, "mm": 0.2},
            "VV": {"ff": 0.7, "aa": 0.2},
            "WW": {"oh": 0.6, "mm": 0.3},
            "YY": {"ih": 0.6, "ee": 0.4},
            "ZZ": {"ss": 0.8, "dd": 0.2}
        }
    
    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.current_state.connected_clients = len(self.active_connections)
        
        # Send initial state
        await self._send_to_client(websocket, self.current_state.dict())
        
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        self.current_state.connected_clients = len(self.active_connections)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def _send_to_client(self, websocket: WebSocket, data: dict):
        """Send data to a specific client with error handling"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self.disconnect(websocket)
    
    async def broadcast_update(self, update: AvatarUpdate):
        """Broadcast avatar update to all connected clients"""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(update.dict())
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def _calculate_interpolated_blend_shapes(
        self, 
        current: Dict[str, float], 
        target: Dict[str, float], 
        interpolation_factor: float
    ) -> Dict[str, float]:
        """Calculate smoothly interpolated values between current and target blend shapes"""
        result = {}
        all_keys = set(current.keys()) | set(target.keys())
        
        for key in all_keys:
            current_val = current.get(key, 0.0)
            target_val = target.get(key, 0.0)
            result[key] = current_val + (target_val - current_val) * interpolation_factor
            
        return result
    
    async def _animation_loop(self):
        """Main animation loop for smooth viseme transitions"""
        try:
            while self.is_animating:
                try:
                    # Get next viseme with timeout
                    viseme_data = await asyncio.wait_for(
                        self.animation_queue.get(), 
                        timeout=0.1
                    )
                    
                    # Calculate transition duration
                    duration = viseme_data.end_time - viseme_data.start_time
                    if duration <= 0:
                        duration = settings.viseme_transition_time
                    
                    # Get target blend shapes
                    target_shapes = self.viseme_to_blend_shape_mapping.get(
                        viseme_data.viseme, 
                        {}
                    )
                    
                    # Animate transition
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        elapsed = time.time() - start_time
                        progress = min(elapsed / duration, 1.0)
                        
                        # Use smooth easing function
                        eased_progress = self._ease_in_out_cubic(progress)
                        
                        # Interpolate blend shapes
                        interpolated = self._calculate_interpolated_blend_shapes(
                            self.current_state.blend_shapes,
                            target_shapes,
                            eased_progress
                        )
                        
                        # Update state
                        self.current_state.blend_shapes = interpolated
                        self.current_state.current_viseme = viseme_data.viseme
                        
                        # Broadcast update
                        update = AvatarUpdate(
                            blend_shapes=interpolated,
                            viseme=viseme_data.viseme,
                            emotion=viseme_data.emotion,
                            timestamp=time.time()
                        )
                        await self.broadcast_update(update)
                        
                        # Small delay for smooth animation
                        await asyncio.sleep(0.016)  # ~60fps
                    
                except asyncio.TimeoutError:
                    # No new viseme, return to neutral if needed
                    if self.current_state.current_viseme != "sil":
                        await self._animate_to_neutral_expression()
                    
        except Exception as e:
            logger.error(f"Animation loop error: {e}")
        finally:
            self.is_animating = False
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth transitions"""
        if t < 0.5:
            return 4 * t * t * t
        p = 2 * t - 2
        return 1 + p * p * p / 2
    
    async def _animate_to_neutral_expression(self):
        """Smoothly animate avatar back to neutral facial expression"""
        viseme_data = VisemeData(
            viseme="sil",
            start_time=time.time(),
            end_time=time.time() + settings.viseme_transition_time,
            blend_shapes={}
        )
        await self.animation_queue.put(viseme_data)
    
    async def add_viseme_sequence(self, sequence: List[VisemeData]):
        """Add a sequence of visemes to the animation queue"""
        for viseme in sequence:
            await self.animation_queue.put(viseme)
        
        # Start animation if not already running
        if not self.is_animating:
            await self.start_animation()
    
    async def start_animation(self):
        """Start the animation loop"""
        if not self.is_animating:
            self.is_animating = True
            self._animation_task = asyncio.create_task(self._animation_loop())
    
    async def stop_animation(self):
        """Stop the animation loop"""
        self.is_animating = False
        if self._animation_task:
            await self._animation_task
        
        # Clear the queue
        while not self.animation_queue.empty():
            try:
                self.animation_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def reset_to_neutral(self):
        """Reset avatar to neutral state"""
        await self.stop_animation()
        
        self.current_state = AvatarState()
        update = AvatarUpdate(
            blend_shapes={},
            viseme="sil",
            emotion=EmotionType.NEUTRAL,
            timestamp=time.time()
        )
        await self.broadcast_update(update)
    
    def get_current_state(self) -> dict:
        """Get current avatar state"""
        return self.current_state.dict()
