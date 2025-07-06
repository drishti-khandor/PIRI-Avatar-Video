"""
OVRLipsync Integration for Real-time Viseme Generation - ENHANCED
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
    ENHANCED OVRLipsync integration for accurate viseme extraction
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
        
        # ENHANCED VRM blend shape mapping for each OVR viseme
        self.ovr_to_vrm_mapping = {
            0: {"Fcl_MTH_Neutral": 0.8, "Fcl_MTH_Close": 0.2},  # Silence
            1: {"Fcl_MTH_Close": 1.0},  # PP (lips together)
            2: {"Fcl_MTH_E": 0.6, "Fcl_MTH_Close": 0.4},        # FF (lip to teeth)
            3: {"Fcl_MTH_E": 0.5, "Fcl_MTH_Small": 0.7},        # TH (tongue visible)
            4: {"Fcl_MTH_E": 0.4, "Fcl_MTH_Small": 0.8},        # DD (tongue to roof)
            5: {"Fcl_MTH_Close": 0.6, "Fcl_MTH_Small": 0.5},    # KK (back tongue)
            6: {"Fcl_MTH_U": 0.6, "Fcl_MTH_Small": 0.9},        # CH (rounded narrow)
            7: {"Fcl_MTH_I": 0.8, "Fcl_MTH_Small": 1.0},        # SS (narrow gap)
            8: {"Fcl_MTH_Close": 0.7, "Fcl_MTH_Neutral": 0.3},  # NN (nasal)
            9: {"Fcl_MTH_U": 0.7, "Fcl_MTH_E": 0.4},           # RR (liquid)
            10: {"Fcl_MTH_A": 1.0, "Fcl_MTH_Large": 0.4},       # AA (open)
            11: {"Fcl_MTH_E": 1.0, "Fcl_MTH_Small": 0.3},       # E (mid)
            12: {"Fcl_MTH_I": 1.0, "Fcl_MTH_Small": 0.6},       # IH (close front)
            13: {"Fcl_MTH_O": 1.0, "Fcl_MTH_U": 0.4},          # OH (mid back)
            14: {"Fcl_MTH_U": 1.0, "Fcl_MTH_O": 0.4}           # OU (close back)
        }
        
        self.sample_rate = 16000  # OVRLipsync expects 16kHz
        
    def convert_audio_format(self, audio_chunk: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """
        ENHANCED audio conversion with better error handling
        """
        try:
            logger.debug(f"🔄 Converting audio: {type(audio_chunk)}, shape: {getattr(audio_chunk, 'shape', 'N/A')}")
            
            # Handle tuple input from TTS
            if isinstance(audio_chunk, tuple):
                original_sample_rate, audio_data = audio_chunk
                audio_chunk = audio_data
                logger.debug(f"📦 Unpacked tuple: rate={original_sample_rate}, data_shape={audio_chunk.shape}")
            
            # Ensure audio is numpy array
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk)
                logger.debug("🔄 Converted to numpy array")
            
            # Convert to float32 if needed
            if audio_chunk.dtype != np.float32:
                if audio_chunk.dtype == np.int16:
                    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                elif audio_chunk.dtype == np.int32:
                    audio_chunk = audio_chunk.astype(np.float32) / 2147483648.0
                else:
                    audio_chunk = audio_chunk.astype(np.float32)
                logger.debug(f"🔄 Converted to float32 from {audio_chunk.dtype}")
            
            # Ensure mono
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
                logger.debug("🔄 Converted to mono")
            
            # Resample to 16kHz if needed
            if original_sample_rate != self.sample_rate:
                import librosa
                audio_chunk = librosa.resample(
                    audio_chunk, 
                    orig_sr=original_sample_rate, 
                    target_sr=self.sample_rate
                )
                logger.debug(f"🔄 Resampled from {original_sample_rate}Hz to {self.sample_rate}Hz")
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = audio_chunk / max_val
                logger.debug(f"🔄 Normalized (max was {max_val:.3f})")
            
            logger.debug(f"✅ Audio conversion complete: {len(audio_chunk)} samples")
            return audio_chunk
            
        except Exception as e:
            logger.error(f"❌ Audio conversion error: {e}")
            # Return silence if conversion fails
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
    
    def extract_visemes_from_audio(self, audio_chunk: np.ndarray, original_sample_rate: int) -> List[OVRViseme]:
        """
        ENHANCED viseme extraction with better logging
        """
        try:
            logger.debug(f"🎵 Starting viseme extraction from audio chunk")
            
            # Convert audio to proper format
            processed_audio = self.convert_audio_format(audio_chunk, original_sample_rate)
            
            if len(processed_audio) == 0:
                logger.warning("❌ Processed audio is empty")
                return [OVRViseme(0, 1.0, 0.0, 0.1)]
            
            logger.debug(f"🎵 Processing {len(processed_audio)} samples")
            
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
                
                logger.debug(f"📁 Created temp WAV file: {temp_wav.name}")
                
                # Run OVRLipsync (this is a placeholder - you'll need the actual OVR binary)
                visemes = self._run_ovr_lipsync(temp_wav.name)
                
                # Clean up
                os.unlink(temp_wav.name)
                
                logger.debug(f"✅ Extracted {len(visemes)} visemes")
                return visemes
                
        except Exception as e:
            logger.error(f"❌ OVR viseme extraction failed: {e}")
            # Return default silence viseme
            duration = len(processed_audio) / self.sample_rate if len(processed_audio) > 0 else 0.1
            return [OVRViseme(0, 1.0, 0.0, duration)]
    
    def _run_ovr_lipsync(self, wav_file_path: str) -> List[OVRViseme]:
        """
        Run OVRLipsync binary and parse results
        Note: This is a placeholder implementation
        """
        try:
            # For now, simulate OVR output with ENHANCED energy-based analysis
            # In production, replace this with actual OVR binary call
            return self._simulate_ovr_output(wav_file_path)
            
        except Exception as e:
            logger.error(f"❌ OVR binary execution failed: {e}")
            return [OVRViseme(0, 1.0, 0.0, 0.1)]
    
    def _simulate_ovr_output(self, wav_file_path: str) -> List[OVRViseme]:
        """
        ENHANCED simulation of OVR output with improved analysis
        """
        try:
            logger.debug(f"🧪 Simulating OVR analysis for: {wav_file_path}")
            
            # Read the WAV file
            with wave.open(wav_file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sample_rate = wav_file.getframerate()
            
            if len(audio_data) == 0:
                return [OVRViseme(0, 1.0, 0.0, 0.1)]
            
            # ENHANCED analysis with better viseme variety
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
                
                # ENHANCED energy and spectral analysis
                energy = np.sum(frame ** 2)
                
                if energy < 1e-6:
                    viseme_id = 0  # Silence
                    weight = 1.0
                else:
                    # Enhanced spectral analysis
                    fft = np.fft.fft(frame)
                    freqs = np.fft.fftfreq(len(frame), 1/sample_rate)
                    magnitude = np.abs(fft)
                    
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
                    dominant_freq = abs(freqs[dominant_freq_idx])
                    
                    # Calculate spectral centroid for better classification
                    spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
                    
                    # ENHANCED frequency to viseme mapping
                    if dominant_freq > 4000:
                        viseme_id = 7  # SS (high freq sibilants)
                    elif dominant_freq > 3000:
                        viseme_id = 6  # CH (fricatives)
                    elif dominant_freq > 2000:
                        if spectral_centroid > 2500:
                            viseme_id = 2  # FF (labiodentals)
                        else:
                            viseme_id = 11  # E (mid vowels)
                    elif dominant_freq > 1500:
                        if energy > 0.01:
                            viseme_id = 10  # AA (open vowels)
                        else:
                            viseme_id = 4  # DD (alveolars)
                    elif dominant_freq > 800:
                        if spectral_centroid > 1200:
                            viseme_id = 12  # IH (close front vowels)
                        else:
                            viseme_id = 13  # OH (mid back vowels)
                    elif dominant_freq > 400:
                        viseme_id = 14  # OU (back vowels)
                    else:
                        if energy > 0.005:
                            viseme_id = 1  # PP (bilabials)
                        else:
                            viseme_id = 8  # NN (nasals)
                    
                    # Scale weight based on energy and confidence
                    weight = min(1.0, energy * 15)  # Increased scaling for better visibility
                    weight = max(0.1, weight)  # Minimum weight for visibility
                
                visemes.append(OVRViseme(viseme_id, weight, start_time, end_time))
                logger.debug(f"🎭 Frame {i//frame_samples}: viseme={viseme_id}, weight={weight:.2f}, freq={dominant_freq:.0f}Hz")
            
            logger.debug(f"✅ Generated {len(visemes)} visemes")
            return visemes if visemes else [OVRViseme(0, 1.0, 0.0, total_duration)]
            
        except Exception as e:
            logger.error(f"❌ Simulated OVR analysis failed: {e}")
            return [OVRViseme(0, 1.0, 0.0, 0.1)]
    
    def visemes_to_blend_shapes(self, visemes: List[OVRViseme]) -> Dict[str, float]:
        """
        ENHANCED conversion of OVR visemes to VRM blend shapes
        """
        try:
            logger.debug(f"🎨 Converting {len(visemes)} visemes to blend shapes")
            
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
                logger.debug("🎨 No visemes, using neutral")
                return blend_shapes
            
            # Find the dominant viseme (highest weight)
            dominant_viseme = max(visemes, key=lambda v: v.weight)
            logger.debug(f"🎯 Dominant viseme: {dominant_viseme.viseme_id} (weight: {dominant_viseme.weight:.2f})")
            
            # Get blend shapes for dominant viseme
            if dominant_viseme.viseme_id in self.ovr_to_vrm_mapping:
                target_shapes = self.ovr_to_vrm_mapping[dominant_viseme.viseme_id]
                
                # Apply weights with boosting for visibility
                for shape_name, base_weight in target_shapes.items():
                    if shape_name in blend_shapes:
                        final_weight = base_weight * dominant_viseme.weight
                        # Boost small weights for better visibility
                        if final_weight > 0.1:
                            final_weight = min(1.0, final_weight * 1.2)
                        blend_shapes[shape_name] = final_weight
                        logger.debug(f"🎨 {shape_name}: {final_weight:.2f}")
            
            # Normalize to prevent conflicts
            normalized = self._normalize_blend_shapes(blend_shapes)
            
            # Log final result
            active_shapes = {k: v for k, v in normalized.items() if v > 0.01}
            logger.debug(f"✅ Final blend shapes: {active_shapes}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Blend shape conversion failed: {e}")
            return {"Fcl_MTH_Neutral": 1.0}
    
    def _normalize_blend_shapes(self, blend_shapes: Dict[str, float]) -> Dict[str, float]:
        """
        ENHANCED normalization to prevent conflicts
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
        
        # Reduce conflicting shapes more aggressively
        if max_weight > 0.05:  # Lower threshold for better exclusivity
            for shape in mouth_shapes:
                if shape != dominant_shape and shape in blend_shapes:
                    blend_shapes[shape] *= (1.0 - max_weight * 0.9)  # More aggressive reduction
        
        # Clamp all values to [0, 1]
        for shape in blend_shapes:
            blend_shapes[shape] = max(0.0, min(1.0, blend_shapes[shape]))
        
        return blend_shapes