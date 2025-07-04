"""
Advanced VRoid Viseme Mapping & Blending System
Implements sophisticated phoneme-to-viseme mapping with smooth transitions
Based on VRoid blend shapes and ARKit-style interpolation
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class BlendShapeWeight:
    """Represents a blend shape and its weight"""
    name: str
    weight: float
    duration: float = 0.1  # How long this weight should be active


@dataclass
class VisemeFrame:
    """A frame of viseme animation with multiple blend shapes"""
    timestamp: float
    blend_shapes: Dict[str, float]
    confidence: float = 1.0


class VisemeTransitionType(Enum):
    """Types of transitions between visemes"""
    LINEAR = "linear"
    SMOOTH = "smooth"
    CUBIC = "cubic"
    ANTICIPATE = "anticipate"


class AdvancedVRoidVisemeMapper:
    """
    Advanced viseme mapping system specifically designed for VRoid models
    with smooth interpolation and natural facial expressions
    """

    def __init__(self):
        # Initialize comprehensive VRoid blend shape mappings
        self.vroid_blend_shapes = self._initialize_vroid_mappings()
        self.phoneme_to_viseme = self._initialize_phoneme_mappings()
        self.transition_curves = self._initialize_transition_curves()
        self.expression_modifiers = self._initialize_expression_modifiers()

        # Animation state
        self.current_frame = 0
        self.target_weights = {}
        self.current_weights = {}
        self.animation_queue = []
        self.is_animating = False

        # Timing controls
        self.frame_rate = 60  # FPS for smooth animation
        self.transition_speed = 0.15  # Seconds for transitions
        self.min_viseme_duration = 0.08  # Minimum time to hold a viseme

    def _initialize_vroid_mappings(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize comprehensive VRoid blend shape mappings for each viseme
        Based on the VRoid blend shapes reference and lip sync best practices
        """
        return {
            # Silence/Rest
            'sil': {
                'Fcl_MTH_Neutral': 1.0,
                'Fcl_MTH_Close': 0.2,
                'Fcl_ALL_Neutral': 0.8
            },

            # Vowel visemes with nuanced mouth shapes
            'aa': {  # Open vowels (AE, AH, EH) - Wide open mouth
                'Fcl_MTH_A': 1.0,
                'Fcl_MTH_Large': 0.3,
                'Fcl_EYE_Natural': 0.9,
                'Fcl_ALL_Neutral': 0.7
            },

            'ah': {  # Open back vowels (AA, AO, AW, OW) - Round open
                'Fcl_MTH_A': 0.8,
                'Fcl_MTH_O': 0.4,
                'Fcl_MTH_Large': 0.5,
                'Fcl_EYE_Natural': 0.9
            },

            'ey': {  # Diphthongs (AY, EY, OY) - Smile-like
                'Fcl_MTH_E': 1.0,
                'Fcl_MTH_Small': 0.3,
                'Fcl_ALL_Joy': 0.2,
                'Fcl_EYE_Joy': 0.3
            },

            'er': {  # R-colored vowels (ER, AX, IX) - Slight pucker
                'Fcl_MTH_E': 0.7,
                'Fcl_MTH_U': 0.3,
                'Fcl_MTH_Small': 0.4
            },

            'ih': {  # Close front vowels (IH, IY) - Narrow opening
                'Fcl_MTH_I': 1.0,
                'Fcl_MTH_Small': 0.6,
                'Fcl_EYE_Natural': 0.9
            },

            'ou': {  # Close back vowels (UW, UH) - Rounded
                'Fcl_MTH_U': 1.0,
                'Fcl_MTH_O': 0.4,
                'Fcl_MTH_Small': 0.3
            },

            # Consonant visemes with realistic mouth positions
            'pp': {  # Bilabials (B, P, M) - Lips together
                'Fcl_MTH_Close': 1.0,
                'Fcl_MTH_Neutral': 0.5,
                'Fcl_EYE_Natural': 0.9
            },

            'ff': {  # Labiodentals (F, V) - Lower lip to teeth
                'Fcl_MTH_E': 0.6,
                'Fcl_MTH_Close': 0.4,
                'Fcl_MTH_Down': 0.3
            },

            'th': {  # Dental fricatives (TH, DH) - Tongue between teeth
                'Fcl_MTH_E': 0.5,
                'Fcl_MTH_Small': 0.7,
                'Fcl_MTH_A': 0.2
            },

            'dd': {  # Alveolars (T, D, N, L) - Tongue to roof
                'Fcl_MTH_E': 0.4,
                'Fcl_MTH_Small': 0.8,
                'Fcl_MTH_Close': 0.3
            },

            'ss': {  # Sibilants (S, Z) - Narrow gap
                'Fcl_MTH_I': 0.8,
                'Fcl_MTH_Small': 1.0,
                'Fcl_MTH_E': 0.3
            },

            'sh': {  # Post-alveolars (SH, ZH, CH, JH) - Rounded narrow
                'Fcl_MTH_U': 0.6,
                'Fcl_MTH_Small': 0.9,
                'Fcl_MTH_O': 0.4
            },

            'kk': {  # Velars (K, G, NG) - Back of tongue
                'Fcl_MTH_Close': 0.6,
                'Fcl_MTH_Small': 0.5,
                'Fcl_MTH_Down': 0.2
            },

            'rr': {  # Approximants (R, W, Y, HH) - Slight rounding
                'Fcl_MTH_U': 0.7,
                'Fcl_MTH_E': 0.4,
                'Fcl_MTH_Small': 0.6
            }
        }

    def _initialize_phoneme_mappings(self) -> Dict[str, str]:
        """Map phonemes to viseme types"""
        return {
            # Silence
            'sil': 'sil', 'sp': 'sil',

            # Vowels
            'AA': 'ah', 'AE': 'aa', 'AH': 'aa', 'AO': 'ah', 'AW': 'ah', 'AX': 'er',
            'AY': 'ey', 'EH': 'aa', 'ER': 'er', 'EY': 'ey', 'IH': 'ih', 'IY': 'ih',
            'IX': 'er', 'OW': 'ah', 'OY': 'ey', 'UH': 'ou', 'UW': 'ou',

            # Consonants
            'B': 'pp', 'CH': 'sh', 'D': 'dd', 'DH': 'th', 'F': 'ff',
            'G': 'kk', 'HH': 'rr', 'JH': 'sh', 'K': 'kk', 'L': 'dd',
            'M': 'pp', 'N': 'dd', 'NG': 'kk', 'P': 'pp', 'R': 'rr',
            'S': 'ss', 'SH': 'sh', 'T': 'dd', 'TH': 'th', 'V': 'ff',
            'W': 'rr', 'Y': 'rr', 'Z': 'ss', 'ZH': 'sh'
        }

    def _initialize_transition_curves(self) -> Dict[str, callable]:
        """Initialize different transition curve functions"""
        return {
            VisemeTransitionType.LINEAR: lambda t: t,
            VisemeTransitionType.SMOOTH: lambda t: t * t * (3.0 - 2.0 * t),  # smoothstep
            VisemeTransitionType.CUBIC: lambda t: t * t * t * (t * (6.0 * t - 15.0) + 10.0),  # smootherstep
            VisemeTransitionType.ANTICIPATE: lambda t: 2.0 * t * t if t < 0.5 else 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
        }

    def _initialize_expression_modifiers(self) -> Dict[str, Dict[str, float]]:
        """Initialize expression modifiers for emotional context"""
        return {
            'happy': {
                'Fcl_ALL_Joy': 0.3,
                'Fcl_EYE_Joy': 0.4,
                'Fcl_MTH_Fun': 0.2
            },
            'sad': {
                'Fcl_ALL_Sorrow': 0.3,
                'Fcl_EYE_Sorrow': 0.4,
                'Fcl_MTH_Down': 0.2
            },
            'surprised': {
                'Fcl_ALL_Surprised': 0.4,
                'Fcl_EYE_Surprised': 0.5,
                'Fcl_MTH_Large': 0.3
            },
            'angry': {
                'Fcl_ALL_Angry': 0.3,
                'Fcl_BRW_Angry': 0.5,
                'Fcl_EYE_Angry': 0.4
            }
        }

    def process_phoneme_sequence(
            self,
            phonemes: List[Tuple[str, float, float]],
            emotion: str = 'neutral',
            transition_type: VisemeTransitionType = VisemeTransitionType.SMOOTH
    ) -> List[VisemeFrame]:
        """
        Process a sequence of phonemes into smooth viseme animation frames

        Args:
            phonemes: List of (phoneme, start_time, end_time) tuples
            emotion: Emotional context modifier
            transition_type: Type of transition between visemes

        Returns:
            List of VisemeFrame objects with blend shape weights
        """
        frames = []

        if not phonemes:
            return [self._create_rest_frame(0.0)]

        # Sort phonemes by start time
        phonemes = sorted(phonemes, key=lambda x: x[1])

        # Calculate total duration
        total_duration = phonemes[-1][2] if phonemes else 1.0
        frame_duration = 1.0 / self.frame_rate

        # Generate frames
        current_time = 0.0
        phoneme_index = 0

        while current_time <= total_duration:
            # Find current and next phonemes
            current_phoneme = self._get_phoneme_at_time(phonemes, current_time)
            next_phoneme = self._get_next_phoneme(phonemes, current_time)

            # Create frame with interpolated weights
            frame = self._create_interpolated_frame(
                current_time,
                current_phoneme,
                next_phoneme,
                emotion,
                transition_type
            )

            frames.append(frame)
            current_time += frame_duration

        return frames

    def _get_phoneme_at_time(self, phonemes: List[Tuple[str, float, float]], time: float) -> Optional[
        Tuple[str, float, float]]:
        """Get the phoneme active at a specific time"""
        for phoneme, start, end in phonemes:
            if start <= time <= end:
                return (phoneme, start, end)
        return None

    def _get_next_phoneme(self, phonemes: List[Tuple[str, float, float]], time: float) -> Optional[
        Tuple[str, float, float]]:
        """Get the next phoneme after a specific time"""
        for phoneme, start, end in phonemes:
            if start > time:
                return (phoneme, start, end)
        return None

    def _create_interpolated_frame(
            self,
            time: float,
            current_phoneme: Optional[Tuple[str, float, float]],
            next_phoneme: Optional[Tuple[str, float, float]],
            emotion: str,
            transition_type: VisemeTransitionType
    ) -> VisemeFrame:
        """Create a frame with interpolated blend shape weights"""

        blend_shapes = {}

        if current_phoneme is None:
            # Rest position
            blend_shapes = self.vroid_blend_shapes.get('sil', {}).copy()
        else:
            phoneme, start, end = current_phoneme
            viseme_type = self.phoneme_to_viseme.get(phoneme.upper(), 'sil')

            # Get base viseme weights
            base_weights = self.vroid_blend_shapes.get(viseme_type, {}).copy()

            # Apply transition if near the end and next phoneme exists
            if next_phoneme and time > (end - self.transition_speed):
                next_phoneme_name, next_start, next_end = next_phoneme
                next_viseme_type = self.phoneme_to_viseme.get(next_phoneme_name.upper(), 'sil')
                next_weights = self.vroid_blend_shapes.get(next_viseme_type, {})

                # Calculate transition progress
                transition_start = end - self.transition_speed
                transition_progress = (time - transition_start) / self.transition_speed
                transition_progress = max(0.0, min(1.0, transition_progress))

                # Apply transition curve
                curve_func = self.transition_curves[transition_type]
                smooth_progress = curve_func(transition_progress)

                # Interpolate between current and next
                blend_shapes = self._interpolate_weights(base_weights, next_weights, smooth_progress)
            else:
                blend_shapes = base_weights

        # Apply emotional modifiers
        if emotion != 'neutral' and emotion in self.expression_modifiers:
            emotion_weights = self.expression_modifiers[emotion]
            for shape_name, weight in emotion_weights.items():
                if shape_name in blend_shapes:
                    blend_shapes[shape_name] = min(1.0, blend_shapes[shape_name] + weight)
                else:
                    blend_shapes[shape_name] = weight

        # Normalize weights to ensure they don't exceed 1.0
        blend_shapes = self._normalize_weights(blend_shapes)

        return VisemeFrame(
            timestamp=time,
            blend_shapes=blend_shapes,
            confidence=1.0
        )

    def _interpolate_weights(
            self,
            weights1: Dict[str, float],
            weights2: Dict[str, float],
            progress: float
    ) -> Dict[str, float]:
        """Interpolate between two sets of blend shape weights"""
        result = {}

        # Get all unique blend shape names
        all_shapes = set(weights1.keys()) | set(weights2.keys())

        for shape in all_shapes:
            w1 = weights1.get(shape, 0.0)
            w2 = weights2.get(shape, 0.0)
            result[shape] = w1 * (1.0 - progress) + w2 * progress

        return result

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to ensure realistic blend shape values"""
        # Clamp all weights to [0, 1] range
        normalized = {}
        for shape, weight in weights.items():
            normalized[shape] = max(0.0, min(1.0, weight))

        # Apply mutual exclusivity rules for conflicting mouth shapes
        mouth_shapes = ['Fcl_MTH_A', 'Fcl_MTH_E', 'Fcl_MTH_I', 'Fcl_MTH_O', 'Fcl_MTH_U', 'Fcl_MTH_Close']
        mouth_weights = {shape: normalized.get(shape, 0.0) for shape in mouth_shapes if shape in normalized}

        if mouth_weights:
            # Find dominant mouth shape and reduce others
            max_shape = max(mouth_weights.keys(), key=lambda k: mouth_weights[k])
            max_weight = mouth_weights[max_shape]

            for shape in mouth_weights:
                if shape != max_shape and shape in normalized:
                    # Reduce conflicting mouth shapes
                    normalized[shape] *= (1.0 - max_weight * 0.7)

        return normalized

    def _create_rest_frame(self, time: float) -> VisemeFrame:
        """Create a rest/neutral frame"""
        return VisemeFrame(
            timestamp=time,
            blend_shapes=self.vroid_blend_shapes['sil'].copy(),
            confidence=1.0
        )

    def create_smooth_animation_sequence(
            self,
            phonemes: List[Tuple[str, float, float]],
            emotion: str = 'neutral'
    ) -> List[Dict[str, float]]:
        """
        Create a smooth animation sequence optimized for real-time playback

        Returns:
            List of blend shape weight dictionaries for each frame
        """
        frames = self.process_phoneme_sequence(phonemes, emotion, VisemeTransitionType.SMOOTH)
        return [frame.blend_shapes for frame in frames]

    def get_instantaneous_viseme_weights(
            self,
            phoneme: str,
            emotion: str = 'neutral'
    ) -> Dict[str, float]:
        """
        Get instantaneous blend shape weights for a single phoneme
        Useful for real-time lip sync without pre-processing
        """
        viseme_type = self.phoneme_to_viseme.get(phoneme.upper(), 'sil')
        weights = self.vroid_blend_shapes.get(viseme_type, {}).copy()

        # Apply emotional modifiers
        if emotion != 'neutral' and emotion in self.expression_modifiers:
            emotion_weights = self.expression_modifiers[emotion]
            for shape_name, weight in emotion_weights.items():
                if shape_name in weights:
                    weights[shape_name] = min(1.0, weights[shape_name] + weight)
                else:
                    weights[shape_name] = weight

        return self._normalize_weights(weights)

    def create_coarticulation_weights(
            self,
            prev_phoneme: str,
            current_phoneme: str,
            next_phoneme: str,
            position_in_phoneme: float = 0.5  # 0.0 = start, 1.0 = end
    ) -> Dict[str, float]:
        """
        Create weights that account for coarticulation (how phonemes influence each other)
        This creates more natural lip sync by considering phoneme context
        """
        # Get base weights for current phoneme
        current_weights = self.get_instantaneous_viseme_weights(current_phoneme)

        # Influence from previous phoneme (stronger at start)
        prev_influence = (1.0 - position_in_phoneme) * 0.3
        if prev_phoneme and prev_influence > 0:
            prev_weights = self.get_instantaneous_viseme_weights(prev_phoneme)
            current_weights = self._interpolate_weights(
                current_weights, prev_weights, prev_influence
            )

        # Anticipation of next phoneme (stronger at end)
        next_influence = position_in_phoneme * 0.2
        if next_phoneme and next_influence > 0:
            next_weights = self.get_instantaneous_viseme_weights(next_phoneme)
            current_weights = self._interpolate_weights(
                current_weights, next_weights, next_influence
            )

        return self._normalize_weights(current_weights)


class VRoidVisemeAnimator:
    """
    Real-time animator for VRoid visemes with smooth transitions
    """

    def __init__(self, mapper: AdvancedVRoidVisemeMapper):
        self.mapper = mapper
        self.current_weights = {}
        self.target_weights = {}
        self.animation_speed = 10.0  # Blend shapes per second
        self.is_running = False
        self.animation_thread = None

    def start_animation_loop(self):
        """Start the animation update loop"""
        if not self.is_running:
            self.is_running = True
            self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self.animation_thread.start()

    def stop_animation_loop(self):
        """Stop the animation update loop"""
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join()

    def set_target_viseme(self, phoneme: str, emotion: str = 'neutral'):
        """Set target viseme weights"""
        self.target_weights = self.mapper.get_instantaneous_viseme_weights(phoneme, emotion)

    def _animation_loop(self):
        """Main animation loop for smooth transitions"""
        frame_time = 1.0 / 60.0  # 60 FPS

        while self.is_running:
            start_time = time.time()

            # Update current weights towards target weights
            self._update_weights(frame_time)

            # Sleep to maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

    def _update_weights(self, delta_time: float):
        """Update current weights towards target weights"""
        blend_speed = self.animation_speed * delta_time

        # Get all unique blend shape names
        all_shapes = set(self.current_weights.keys()) | set(self.target_weights.keys())

        for shape in all_shapes:
            current = self.current_weights.get(shape, 0.0)
            target = self.target_weights.get(shape, 0.0)

            # Smooth interpolation towards target
            diff = target - current
            if abs(diff) < 0.001:
                self.current_weights[shape] = target
            else:
                self.current_weights[shape] = current + diff * min(1.0, blend_speed)

    def get_current_weights(self) -> Dict[str, float]:
        """Get current blend shape weights"""
        return self.current_weights.copy()


# Example usage and testing
if __name__ == "__main__":
    # Create the advanced viseme mapper
    mapper = AdvancedVRoidVisemeMapper()

    # Test with a sample phoneme sequence
    phonemes = [
        ('HH', 0.0, 0.1),  # Hello
        ('AH', 0.1, 0.3),
        ('L', 0.3, 0.4),
        ('OW', 0.4, 0.6),
        ('W', 0.7, 0.8),  # World
        ('ER', 0.8, 1.0),
        ('L', 1.0, 1.1),
        ('D', 1.1, 1.2)
    ]

    # Generate smooth animation sequence
    animation_frames = mapper.create_smooth_animation_sequence(phonemes, emotion='happy')

    print(f"Generated {len(animation_frames)} animation frames")
    print("Sample frame:", animation_frames[10] if len(animation_frames) > 10 else animation_frames[0])

    # Test real-time viseme weights
    weights = mapper.get_instantaneous_viseme_weights('AH', 'surprised')
    print("Instantaneous 'AH' weights:", weights)

    # Test coarticulation
    coart_weights = mapper.create_coarticulation_weights('P', 'AH', 'T', 0.5)
    print("Coarticulation weights:", coart_weights)