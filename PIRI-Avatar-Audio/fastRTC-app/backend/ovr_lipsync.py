"""
OVRLipsync Integration for Real-time Viseme Generation
Replaces the existing viseme extraction with Meta's OVRLipsync
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import subprocess
import tempfile
import os
import json
import wave

logger = logging.getLogger(__name__)

@dataclass
class OVRViseme:
    """OVR Viseme data structure"""
    viseme_id: int
    weight: float
    start_time: float
    end_time: float

class OVRLipsyncExtractor:
    """
    OVRLipsync integration for accurate viseme extraction
    """
    
    def __init__(self):
        # OVR Viseme mapping (0-14 standard)
        self.ovr_viseme_names = {
            0: "sil",      # Silence
            1: "PP",       # Bilabials (P, B, M)
            2: "FF",       # Labiodentals (F, V)
            3: "TH",       # Dental fricatives (TH, DH)
            4: "DD",       # Alveolars (T, D, N, L)
            5: "kk",       # Velars (K, G, NG)
            6: "CH",       # Post-alveolars (CH, JH, SH, ZH)
            7: "SS",       # Sibilants (S, Z)
            8: "nn",       # Nasals (N, M, NG)
            9: "RR",       # Liquids (R, L)
            10: "aa",      # Open vowels (AA, AE, AH)
            11: "E",       # Mid vowels (EH, ER, AX)
            12: "ih",      # Close front vowels (IH, IY)
            13: "oh",      # Mid back vowels (AO, OW)
            14: "ou"       # Close back vowels (UW, UH)
        }
        
        # VRM blend shape mapping for each OVR viseme
        self.ovr_to_vrm_mapping = {
            0: {"Fcl_MTH_Neutral": 1.0, "Fcl_MTH_Close": 0.3},  # Silence
            1: {"Fcl_MTH_Close": 1.0, "Fcl_MTH_Neutral": 0.2},  # PP (lips together)
            2: {"Fcl_MTH_E": 0.6, "Fcl_MTH_Close": 0.4},        # FF (lip to teeth)
            3: {"Fcl_MTH_E": 0.5, "Fcl_MTH_Small": 0.7},        # TH (tongue visible)
            4: {"Fcl_MTH_E": 0.4, "Fcl_MTH_Small": 0.8},        # DD (tongue to roof)
            5: {"Fcl_MTH_Close": 0.6, "Fcl_MTH_Small": 0.5},    # KK (back tongue)
            6: {"Fcl_MTH_U": 0.6, "Fcl_MTH_Small": 0.9},        # CH (rounded narrow)
            7: {"Fcl_MTH_I": 0.8, "Fcl_MTH_Small": 1.0},        # SS (narrow gap)
            8: {"Fcl_MTH_Close": 0.7, "Fcl_MTH_Neutral": 0.4},  # NN (nasal)
            9: {"Fcl_MTH_U": 0.7, "Fcl_MTH_E": 0.4},           # RR (liquid)
            10: {"Fcl_MTH_A": 1.0, "Fcl_MTH_Large": 0.3},       # AA (open)
            11: {"Fcl_MTH_E": 1.0, "Fcl_MTH_Small": 0.3},       # E (mid)
            12: {"Fcl_MTH_I": 1.0, "Fcl_MTH_Small": 0.6},       # IH (close front)
            13: {"Fcl_MTH_O": 1.0, "Fcl_MTH_U": 0.4},          # OH (mid back)
            14: {"Fcl_MTH_U": 1.0, "Fcl_MTH_O": 0.4}           # OU (close back)
        }
        
        self.sample_rate = 16000  # OVRLipsync expects 16kHz
        
    def convert_audio_format(self, audio_chunk: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """
        Convert audio to OVRLipsync compatible format
        Input: (sample_rate, audio_data) tuple from TTS
        Output: 16kHz mono float32 audio
        """
        try:
            # Handle tuple input from TTS
            if isinstance(audio_chunk, tuple):
                original_sample_rate, audio_data = audio_chunk
                audio_chunk = audio_data
            
            # Ensure audio is numpy array
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk)
            
            # Convert to float32 if needed
            if audio_chunk.dtype != np.float32:
                if audio_chunk.dtype == np.int16:
                    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                elif audio_chunk.dtype == np.int32:
                    audio_chunk = audio_chunk.astype(np.float32) / 2147483648.0
                else:
                    audio_chunk = audio_chunk.astype(np.float32)
            
            # Ensure mono
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Resample to 16kHz if needed
            if original_sample_rate != self.sample_rate:
                import librosa
                audio_chunk = librosa.resample(
                    audio_chunk, 
                    orig_sr=original_sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = audio_chunk / max_val
            
            return audio_chunk
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            # Return silence if conversion fails
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
    
    def extract_visemes_from_audio(self, audio_chunk: np.ndarray, original_sample_rate: int) -> List[OVRViseme]:
        """
        Extract visemes using OVRLipsync
        """
        try:
            # Convert audio to proper format
            processed_audio = self.convert_audio_format(audio_chunk, original_sample_rate)
            
            if len(processed_audio) == 0:
                return [OVRViseme(0, 1.0, 0.0, 0.1)]
            
            # Create temporary WAV file for OVRLipsync
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                # Write WAV file
                with wave.open(temp_wav.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    
                    # Convert float32 to int16 for WAV
                    audio_int16 = (processed_audio * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Run OVRLipsync (this is a placeholder - you'll need the actual OVR binary)
                visemes = self._run_ovr_lipsync(temp_wav.name)
                
                # Clean up
                os.unlink(temp_wav.name)
                
                return visemes
                
        except Exception as e:
            logger.error(f"OVR viseme extraction failed: {e}")
            # Return default silence viseme
            duration = len(processed_audio) / self.sample_rate if len(processed_audio) > 0 else 0.1
            return [OVRViseme(0, 1.0, 0.0, duration)]
    
    def _run_ovr_lipsync(self, wav_file_path: str) -> List[OVRViseme]:
        """
        Run OVRLipsync binary and parse results
        Note: This is a placeholder implementation
        """
        try:
            # For now, simulate OVR output with energy-based analysis
            # In production, replace this with actual OVR binary call
            return self._simulate_ovr_output(wav_file_path)
            
        except Exception as e:
            logger.error(f"OVR binary execution failed: {e}")
            return [OVRViseme(0, 1.0, 0.0, 0.1)]
    
    def _simulate_ovr_output(self, wav_file_path: str) -> List[OVRViseme]:
        """
        Simulate OVR output with improved energy-based analysis
        Replace this with actual OVR integration
        """
        try:
            # Read the WAV file
            with wave.open(wav_file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sample_rate = wav_file.getframerate()
            
            if len(audio_data) == 0:
                return [OVRViseme(0, 1.0, 0.0, 0.1)]
            
            # Analyze audio in frames
            frame_duration = 0.02  # 20ms frames
            frame_samples = int(frame_duration * sample_rate)
            total_duration = len(audio_data) / sample_rate
            
            visemes = []
            
            for i in range(0, len(audio_data), frame_samples):
                frame = audio_data[i:i + frame_samples]
                start_time = i / sample_rate
                end_time = min((i + frame_samples) / sample_rate, total_duration)
                
                if len(frame) == 0:
                    continue
                
                # Energy-based viseme selection (improved)
                energy = np.sum(frame ** 2)
                
                if energy < 1e-6:
                    viseme_id = 0  # Silence
                    weight = 1.0
                else:
                    # Spectral analysis for better viseme classification
                    fft = np.fft.fft(frame)
                    freqs = np.fft.fftfreq(len(frame), 1/sample_rate)
                    magnitude = np.abs(fft)
                    
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
                    dominant_freq = abs(freqs[dominant_freq_idx])
                    
                    # Map frequency to viseme (simplified)
                    if dominant_freq > 3000:
                        viseme_id = 7  # SS (high freq sibilants)
                    elif dominant_freq > 2000:
                        viseme_id = 6  # CH (fricatives)
                    elif dominant_freq > 1500:
                        viseme_id = 11  # E (mid vowels)
                    elif dominant_freq > 800:
                        viseme_id = 10  # AA (open vowels)
                    elif dominant_freq > 400:
                        viseme_id = 14  # OU (back vowels)
                    else:
                        viseme_id = 1  # PP (low freq consonants)
                    
                    weight = min(1.0, energy * 10)  # Scale energy to weight
                
                visemes.append(OVRViseme(viseme_id, weight, start_time, end_time))
            
            return visemes if visemes else [OVRViseme(0, 1.0, 0.0, total_duration)]
            
        except Exception as e:
            logger.error(f"Simulated OVR analysis failed: {e}")
            return [OVRViseme(0, 1.0, 0.0, 0.1)]
    
    def visemes_to_blend_shapes(self, visemes: List[OVRViseme]) -> Dict[str, float]:
        """
        Convert OVR visemes to VRM blend shapes with proper weighting
        """
        try:
            # Initialize all blend shapes to 0
            blend_shapes = {
                "Fcl_MTH_Neutral": 0.0,
                "Fcl_MTH_A": 0.0,
                "Fcl_MTH_E": 0.0,
                "Fcl_MTH_I": 0.0,
                "Fcl_MTH_O": 0.0,
                "Fcl_MTH_U": 0.0,
                "Fcl_MTH_Close": 0.0,
                "Fcl_MTH_Small": 0.0,
                "Fcl_MTH_Large": 0.0
            }
            
            if not visemes:
                blend_shapes["Fcl_MTH_Neutral"] = 1.0
                return blend_shapes
            
            # Find the dominant viseme
            dominant_viseme = max(visemes, key=lambda v: v.weight)
            
            # Get blend shapes for dominant viseme
            if dominant_viseme.viseme_id in self.ovr_to_vrm_mapping:
                target_shapes = self.ovr_to_vrm_mapping[dominant_viseme.viseme_id]
                
                # Apply weights
                for shape_name, base_weight in target_shapes.items():
                    if shape_name in blend_shapes:
                        blend_shapes[shape_name] = base_weight * dominant_viseme.weight
            
            # Normalize to prevent conflicts
            return self._normalize_blend_shapes(blend_shapes)
            
        except Exception as e:
            logger.error(f"Blend shape conversion failed: {e}")
            return {"Fcl_MTH_Neutral": 1.0}
    
    def _normalize_blend_shapes(self, blend_shapes: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize blend shapes to prevent conflicts
        """
        # Ensure mutual exclusivity for mouth shapes
        mouth_shapes = ["Fcl_MTH_A", "Fcl_MTH_E", "Fcl_MTH_I", "Fcl_MTH_O", "Fcl_MTH_U", "Fcl_MTH_Close"]
        
        # Find the dominant mouth shape
        max_weight = 0
        dominant_shape = "Fcl_MTH_Neutral"
        
        for shape in mouth_shapes:
            if shape in blend_shapes and blend_shapes[shape] > max_weight:
                max_weight = blend_shapes[shape]
                dominant_shape = shape
        
        # Reduce conflicting shapes
        if max_weight > 0.1:
            for shape in mouth_shapes:
                if shape != dominant_shape and shape in blend_shapes:
                    blend_shapes[shape] *= (1.0 - max_weight * 0.8)
        
        # Clamp all values to [0, 1]
        for shape in blend_shapes:
            blend_shapes[shape] = max(0.0, min(1.0, blend_shapes[shape]))
        
        return blend_shapes