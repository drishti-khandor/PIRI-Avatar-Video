"""
VRoid Viseme Mapper
Simple mapping between visemes and VRoid blend shapes based on the Microsoft Speech Platform viseme IDs.
Uses only the exact blend shape keys and weights specified in the mapping table.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VisemeFrame:
    """A frame of viseme animation with blend shapes"""
    timestamp: float
    blend_shapes: Dict[str, float]
    confidence: float = 1.0

class VRoidVisemeMapper:
    """
    Simple VRoid viseme mapper that converts viseme IDs to VRoid blend shapes
    Based on the exact specification provided in the mapping table.
    """
    
    def __init__(self):
        """Initialize the VRoid viseme mapper with the exact blend shape mappings"""
        self.viseme_to_blend_shapes = self._initialize_viseme_mappings()
        logger.info("VRoid viseme mapper initialized")
    
    def _initialize_viseme_mappings(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize VRoid blend shape mappings for each viseme ID
        Based on the exact specification:
        
        viseme ID | Typical phoneme | VRoid shape-key(s) to drive | Suggested max weight
        0         | silence         | —                           | clamp everything ≤ 0.01
        1         | p / b / m       | FcL_MTH_Close (lip press)   | 0.80
        2         | f / v           | FcL_MTH_Small + FcL_MTH_Up  | 0.65
        3         | th              | FcL_MTH_Small + FcL_MTH_Down| 0.65
        4         | t / d / l       | FcL_MTH_Small (optional tongue) | 0.50
        5         | k / g / ng      | FcL_MTH_Small (jaw drop a bit) | 0.55
        6         | ch / sh / jh    | FcL_MTH_Close + Small (blend 50/50) | 0.70
        7         | s / z           | FcL_MTH_Small               | 0.60
        8         | n               | FcL_MTH_Close (so lips touch) | 0.80
        9         | r               | FcL_MTH_Neutral or leave at last vowel | n/a
        10        | "a" (ah)        | FcL_MTH_A                   | 1.00
        11        | "eh"            | FcL_MTH_E                   | 1.00
        12        | "ee"            | FcL_MTH_I                   | 1.00
        13        | "oh"            | FcL_MTH_O                   | 1.00
        14        | "oo"            | FcL_MTH_U                   | 1.00
        """
        return {
            # ID 0: silence - clamp everything ≤ 0.01
            'sil': {},  # Empty dict means all weights clamped to 0.01 or less
            
            # ID 1: p/b/m - FcL_MTH_Close (lip press) - 0.80
            'pp': {
                'Fcl_MTH_Close': 0.80
            },
            
            # ID 2: f/v - FcL_MTH_Small + FcL_MTH_Up - 0.65
            'ff': {
                'Fcl_MTH_Small': 0.65,
                'Fcl_MTH_Up': 0.65
            },
            
            # ID 3: th - FcL_MTH_Small + FcL_MTH_Down - 0.65
            'th': {
                'Fcl_MTH_Small': 0.65,
                'Fcl_MTH_Down': 0.65
            },
            
            # ID 4: t/d/l - FcL_MTH_Small (optional tongue) - 0.50
            'dd': {
                'Fcl_MTH_Small': 0.50
            },
            
            # ID 5: k/g/ng - FcL_MTH_Small (jaw drop a bit) - 0.55
            'kk': {
                'Fcl_MTH_Small': 0.55
            },
            
            # ID 6: ch/sh/jh - FcL_MTH_Close + Small (blend 50/50) - 0.70
            'ch': {
                'Fcl_MTH_Close': 0.35,  # 50% of 0.70
                'Fcl_MTH_Small': 0.35   # 50% of 0.70
            },
            
            # ID 7: s/z - FcL_MTH_Small - 0.60
            'ss': {
                'Fcl_MTH_Small': 0.60
            },
            
            # ID 8: n - FcL_MTH_Close (so lips touch) - 0.80
            'nn': {
                'Fcl_MTH_Close': 0.80
            },
            
            # ID 9: r - FcL_MTH_Neutral or leave at last vowel - n/a
            'rr': {
                'Fcl_MTH_Neutral': 0.50
            },
            
            # ID 10: "a" (ah) - FcL_MTH_A - 1.00
            'aa': {
                'Fcl_MTH_A': 1.00
            },
            
            # ID 11: "eh" - FcL_MTH_E - 1.00
            'eh': {
                'Fcl_MTH_E': 1.00
            },
            
            # ID 12: "ee" - FcL_MTH_I - 1.00
            'ih': {
                'Fcl_MTH_I': 1.00
            },
            
            # ID 13: "oh" - FcL_MTH_O - 1.00
            'oh': {
                'Fcl_MTH_O': 1.00
            },
            
            # ID 14: "oo" - FcL_MTH_U - 1.00
            'oo': {
                'Fcl_MTH_U': 1.00
            }
        }
    
    def get_instantaneous_viseme_weights(self, viseme: str) -> Dict[str, float]:
        """
        Get instantaneous blend shape weights for a single viseme.
        
        Args:
            viseme: Viseme string (sil, pp, ff, th, dd, kk, ch, ss, nn, rr, aa, eh, ih, oh, oo)
            
        Returns:
            Dictionary of blend shape names and weights
        """
        # Get weights for the viseme, default to silence if not found
        weights = self.viseme_to_blend_shapes.get(viseme, {}).copy()
        
        logger.info(f"Getting weights for viseme '{viseme}': {weights}")
        
        # For silence, clamp all possible blend shapes to ≤ 0.01
        if viseme == 'sil' or not weights:
            all_possible_shapes = [
                'Fcl_MTH_Close', 'Fcl_MTH_Small', 'Fcl_MTH_Up', 'Fcl_MTH_Down',
                'Fcl_MTH_Neutral', 'Fcl_MTH_A', 'Fcl_MTH_E', 'Fcl_MTH_I',
                'Fcl_MTH_O', 'Fcl_MTH_U'
            ]
            weights = {shape: 0.01 for shape in all_possible_shapes}
        
        return weights
    
    def create_animation_frames(
        self,
        visemes: List[tuple],  # [(viseme, start_time, end_time), ...]
        frame_rate: float = 60.0
    ) -> List[VisemeFrame]:
        """
        Create animation frames from a sequence of visemes.
        
        Args:
            visemes: List of (viseme, start_time, end_time) tuples
            frame_rate: Animation frame rate (fps)
            
        Returns:
            List of VisemeFrame objects with blend shape weights
        """
        frames = []
        
        if not visemes:
            return [self._create_rest_frame(0.0)]
        
        # Sort visemes by start time
        visemes = sorted(visemes, key=lambda x: x[1])
        
        # Calculate total duration
        total_duration = visemes[-1][2] if visemes else 1.0
        frame_duration = 1.0 / frame_rate
        
        # Generate frames
        current_time = 0.0
        
        while current_time <= total_duration:
            # Find current viseme
            current_viseme = self._get_viseme_at_time(visemes, current_time)
            
            if current_viseme:
                viseme, start, end = current_viseme
                weights = self.get_instantaneous_viseme_weights(viseme)
            else:
                weights = self.get_instantaneous_viseme_weights('sil')
            
            frame = VisemeFrame(
                timestamp=current_time,
                blend_shapes=weights,
                confidence=1.0
            )
            frames.append(frame)
            
            current_time += frame_duration
        
        return frames
    
    def _get_viseme_at_time(self, visemes: List[tuple], time: float) -> Optional[tuple]:
        """Get the viseme active at a specific time"""
        for viseme, start, end in visemes:
            if start <= time <= end:
                return (viseme, start, end)
        return None
    
    def _create_rest_frame(self, time: float) -> VisemeFrame:
        """Create a rest/neutral frame"""
        return VisemeFrame(
            timestamp=time,
            blend_shapes=self.get_instantaneous_viseme_weights('sil'),
            confidence=1.0
        )
    
    def get_available_visemes(self) -> List[str]:
        """
        Get list of available visemes.
        
        Returns:
            List of viseme strings
        """
        return list(self.viseme_to_blend_shapes.keys())


# Example usage and testing
if __name__ == "__main__":
    # Create the VRoid viseme mapper
    mapper = VRoidVisemeMapper()
    
    # Test viseme weights
    print("Available visemes:", mapper.get_available_visemes())
    
    # Test individual viseme weights
    test_visemes = ['sil', 'pp', 'aa', 'ih', 'oo', 'ff', 'th']
    for viseme in test_visemes:
        weights = mapper.get_instantaneous_viseme_weights(viseme)
        print(f"Viseme '{viseme}' weights:", weights)
