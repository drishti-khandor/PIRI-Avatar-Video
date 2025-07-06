"""
Smooth Animation System with Exponential Moving Average
Provides lag-free, smooth transitions for VRM blend shapes
"""

import numpy as np
import time
import asyncio
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnimationFrame:
    """Single animation frame with blend shapes and timing"""
    timestamp: float
    blend_shapes: Dict[str, float]
    duration: float = 0.016  # ~60fps

class SmoothAnimator:
    """
    Exponential Moving Average based smooth animator
    Eliminates jitter and provides natural transitions
    """
    
    def __init__(self, smoothing_factor: float = 0.15, target_fps: float = 60.0):
        self.smoothing_factor = smoothing_factor  # Lower = smoother, higher = more responsive
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        
        # Current state
        self.current_blend_shapes = {}
        self.target_blend_shapes = {}
        
        # Animation control
        self.is_animating = False
        self.animation_thread = None
        self.should_stop = False
        
        # Callbacks
        self.on_frame_callback = None
        
        # Performance tracking
        self.last_frame_time = 0
        self.frame_count = 0
        
    def set_frame_callback(self, callback):
        """Set callback function to receive animation frames"""
        self.on_frame_callback = callback
    
    def start_animation(self):
        """Start the smooth animation loop"""
        if not self.is_animating:
            self.is_animating = True
            self.should_stop = False
            self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self.animation_thread.start()
            logger.info("Smooth animator started")
    
    def stop_animation(self):
        """Stop the animation loop"""
        self.should_stop = True
        self.is_animating = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)
        logger.info("Smooth animator stopped")
    
    def update_target(self, new_blend_shapes: Dict[str, float]):
        """
        Update target blend shapes for smooth transition
        """
        try:
            # Initialize current shapes if empty
            if not self.current_blend_shapes:
                self.current_blend_shapes = {k: 0.0 for k in new_blend_shapes.keys()}
            
            # Update targets
            self.target_blend_shapes = new_blend_shapes.copy()
            
            # Ensure all shapes exist in current state
            for shape_name in new_blend_shapes:
                if shape_name not in self.current_blend_shapes:
                    self.current_blend_shapes[shape_name] = 0.0
            
            logger.debug(f"Updated animation targets: {len(new_blend_shapes)} shapes")
            
        except Exception as e:
            logger.error(f"Failed to update animation target: {e}")
    
    def _animation_loop(self):
        """Main animation loop with precise timing"""
        logger.info("Animation loop started")
        
        while not self.should_stop:
            frame_start = time.time()
            
            try:
                # Update blend shapes using exponential moving average
                self._update_blend_shapes()
                
                # Create animation frame
                frame = AnimationFrame(
                    timestamp=frame_start,
                    blend_shapes=self.current_blend_shapes.copy(),
                    duration=self.frame_duration
                )
                
                # Send frame to callback
                if self.on_frame_callback:
                    try:
                        self.on_frame_callback(frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                
                # Performance tracking
                self.frame_count += 1
                if self.frame_count % 300 == 0:  # Log every 5 seconds at 60fps
                    logger.debug(f"Animation running: {self.frame_count} frames processed")
                
            except Exception as e:
                logger.error(f"Animation loop error: {e}")
            
            # Precise frame timing
            frame_time = time.time() - frame_start
            sleep_time = max(0, self.frame_duration - frame_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif frame_time > self.frame_duration * 1.5:
                logger.warning(f"Frame took too long: {frame_time:.3f}s (target: {self.frame_duration:.3f}s)")
        
        logger.info("Animation loop stopped")
    
    def _update_blend_shapes(self):
        """
        Update current blend shapes using exponential moving average
        """
        try:
            if not self.target_blend_shapes:
                return
            
            # Exponential moving average for each blend shape
            for shape_name, target_value in self.target_blend_shapes.items():
                if shape_name in self.current_blend_shapes:
                    current_value = self.current_blend_shapes[shape_name]
                    
                    # EMA formula: new_value = α * target + (1 - α) * current
                    new_value = (self.smoothing_factor * target_value + 
                               (1 - self.smoothing_factor) * current_value)
                    
                    self.current_blend_shapes[shape_name] = new_value
                else:
                    # Initialize new shape
                    self.current_blend_shapes[shape_name] = target_value * self.smoothing_factor
            
            # Decay unused shapes
            for shape_name in list(self.current_blend_shapes.keys()):
                if shape_name not in self.target_blend_shapes:
                    self.current_blend_shapes[shape_name] *= (1 - self.smoothing_factor)
                    
                    # Remove very small values
                    if self.current_blend_shapes[shape_name] < 0.001:
                        self.current_blend_shapes[shape_name] = 0.0
            
        except Exception as e:
            logger.error(f"Blend shape update error: {e}")
    
    def get_current_state(self) -> Dict[str, float]:
        """Get current blend shape state"""
        return self.current_blend_shapes.copy()
    
    def set_smoothing_factor(self, factor: float):
        """
        Adjust smoothing factor
        0.0 = maximum smoothing (slow response)
        1.0 = no smoothing (immediate response)
        """
        self.smoothing_factor = max(0.0, min(1.0, factor))
        logger.info(f"Smoothing factor set to: {self.smoothing_factor}")
    
    def reset_to_neutral(self):
        """Reset to neutral expression"""
        neutral_shapes = {
            "Fcl_MTH_Neutral": 1.0,
            "Fcl_MTH_A": 0.0,
            "Fcl_MTH_E": 0.0,
            "Fcl_MTH_I": 0.0,
            "Fcl_MTH_O": 0.0,
            "Fcl_MTH_U": 0.0,
            "Fcl_MTH_Close": 0.0,
            "Fcl_MTH_Small": 0.0,
            "Fcl_MTH_Large": 0.0
        }
        self.update_target(neutral_shapes)
        logger.info("Reset to neutral expression")

class FrameBuffer:
    """
    Buffer for managing animation frames with timing
    """
    
    def __init__(self, max_buffer_size: int = 300):  # 5 seconds at 60fps
        self.frames: List[AnimationFrame] = []
        self.max_size = max_buffer_size
        self.current_index = 0
        
    def add_frame(self, frame: AnimationFrame):
        """Add frame to buffer"""
        self.frames.append(frame)
        
        # Remove old frames if buffer is full
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
    
    def get_frame_at_time(self, timestamp: float) -> Optional[AnimationFrame]:
        """Get frame closest to specified timestamp"""
        if not self.frames:
            return None
        
        # Find closest frame
        closest_frame = min(self.frames, key=lambda f: abs(f.timestamp - timestamp))
        return closest_frame
    
    def clear(self):
        """Clear all frames"""
        self.frames.clear()
        self.current_index = 0