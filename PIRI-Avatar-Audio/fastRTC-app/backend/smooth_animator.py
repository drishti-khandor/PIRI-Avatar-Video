"""
Smooth Animation System — now actually thread-safe & VRM-1.0-aware
------------------------------------------------------------------
• Exponential-moving-average smoothing (configurable).
• Monotonic timing via time.perf_counter().
• Hard mutex so the background thread never fights the main thread.
• Automatic garbage-collection of dead keys.
"""

import time
import threading
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnimationFrame:
    """Single animation frame with blend shapes and timing."""
    timestamp: float
    blend_shapes: Dict[str, float]
    duration: float = 0.016  # ~60 fps


class SmoothAnimator:
    """
    EMA-based animator that eliminates jitter without introducing Hitler-stache lag.
    """
    def __init__(self, smoothing_factor: float = 0.15, target_fps: float = 60.0):
        self.smoothing_factor = smoothing_factor          # 0 = butter, 1 = raw
        self.frame_duration = 1.0 / target_fps

        # shared state (protected by _lock)
        self.current_blend_shapes: Dict[str, float] = {}
        self.target_blend_shapes: Dict[str, float] = {}

        # threading
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # callbacks
        self.on_frame_callback = None

        # stats
        self.frame_count = 0

    # ───────────────────────────────────────── public API ──────────────────────────────────────────

    def set_frame_callback(self, callback):
        self.on_frame_callback = callback

    def start_animation(self):
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("SmoothAnimator ▶ started")

    def stop_animation(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("SmoothAnimator ■ stopped")

    def update_target(self, new_blend_shapes: Dict[str, float]):
        """Push a new target dict; thread-safe."""
        with self._lock:
            if not self.current_blend_shapes:
                self.current_blend_shapes = {k: 0.0 for k in new_blend_shapes}
            self.target_blend_shapes = new_blend_shapes.copy()
            # seed unseen keys in current state
            for k in new_blend_shapes:
                self.current_blend_shapes.setdefault(k, 0.0)

    def get_current_state(self) -> Dict[str, float]:
        with self._lock:
            return self.current_blend_shapes.copy()

    def set_smoothing_factor(self, factor: float):
        self.smoothing_factor = max(0.0, min(1.0, factor))
        logger.info(f"Smoothing factor → {self.smoothing_factor}")

    def reset_to_neutral(self):
        """Zero out everything (VRM-1.0 vowels only)."""
        self.update_target({"aa": 0.0, "ih": 0.0, "ou": 0.0})
        logger.info("Animator reset → neutral")

    # ───────────────────────────────────────── internal ───────────────────────────────────────────

    def _loop(self):
        logger.debug("Animation thread up")
        while not self._stop_flag.is_set():
            frame_start = time.perf_counter()

            # physics step
            with self._lock:
                self._ema_step()

                frame = AnimationFrame(
                    timestamp=frame_start,
                    blend_shapes=self.current_blend_shapes.copy(),
                    duration=self.frame_duration
                )

            # callback (outside lock – don’t block the physics)
            try:
                if self.on_frame_callback:
                    self.on_frame_callback(frame)
            except Exception as exc:
                logger.error(f"Frame callback blew up: {exc}")

            # stats & pacing
            self.frame_count += 1
            if self.frame_count % 300 == 0:
                logger.debug(f"Animator alive — {self.frame_count} frames")

            elapsed = time.perf_counter() - frame_start
            sleep_for = max(0.0, self.frame_duration - elapsed)
            if sleep_for:
                time.sleep(sleep_for)
            elif elapsed > self.frame_duration * 1.5:
                logger.warning(f"Frame overrun: {elapsed:.3f}s (target {self.frame_duration:.3f}s)")

        logger.debug("Animation thread down")

    def _ema_step(self):
        """Single EMA iteration; assumes caller holds _lock."""
        if not self.target_blend_shapes:
            return

        α = self.smoothing_factor
        for shape, target in self.target_blend_shapes.items():
            cur = self.current_blend_shapes.get(shape, 0.0)
            self.current_blend_shapes[shape] = (α * target) + ((1.0 - α) * cur)

        # exponential decay of keys not in target
        for shape in list(self.current_blend_shapes.keys()):
            if shape not in self.target_blend_shapes:
                self.current_blend_shapes[shape] *= (1.0 - α)
                if self.current_blend_shapes[shape] < 0.001:
                    del self.current_blend_shapes[shape]


# ───────────────────────────────────────── simple ring buffer ─────────────────────────────────────

class FrameBuffer:
    """Tiny ring buffer for recorded frames — useful for debugging or lag-compensation."""
    def __init__(self, max_buffer_size: int = 300):
        self.frames: List[AnimationFrame] = []
        self.max_size = max_buffer_size

    def add_frame(self, frame: AnimationFrame):
        self.frames.append(frame)
        if len(self.frames) > self.max_size:
            self.frames.pop(0)

    def get_frame_at_time(self, timestamp: float) -> Optional[AnimationFrame]:
        if not self.frames:
            return None
        return min(self.frames, key=lambda f: abs(f.timestamp - timestamp))

    def clear(self):
        self.frames.clear()