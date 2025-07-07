"""
Real OVRLipsync Integration for Viseme Generation
Uses actual OVR Lipsync library with proper logging
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

# Import the actual OVR Lipsync library
try:
    import ovr_lipsync
    OVR_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ OVR Lipsync library imported successfully")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Failed to import OVR Lipsync: {e}")
    logger.error("Install with: pip install ovr-lipsync")
    OVR_AVAILABLE = False

@dataclass
class OVRViseme:
    """OVR Viseme data structure"""
    viseme_id: int
    weight: float
    start_time: float
    end_time: float

class OVRLipsyncExtractor:
    """
    Real OVRLipsync integration using actual library functions
    """

    def __init__(self):
        logger.info("🔧 Initializing OVRLipsyncExtractor")

        if not OVR_AVAILABLE:
            logger.error("❌ OVR Lipsync library not available")
            raise ImportError("OVR Lipsync library not installed")

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
            0: {"Fcl_MTH_Neutral": 0.8, "Fcl_MTH_Close": 0.2},
            1: {"Fcl_MTH_Close": 1.0},
            2: {"Fcl_MTH_E": 0.6, "Fcl_MTH_Close": 0.4},
            3: {"Fcl_MTH_E": 0.5, "Fcl_MTH_Small": 0.7},
            4: {"Fcl_MTH_E": 0.4, "Fcl_MTH_Small": 0.8},
            5: {"Fcl_MTH_Close": 0.6, "Fcl_MTH_Small": 0.5},
            6: {"Fcl_MTH_U": 0.6, "Fcl_MTH_Small": 0.9},
            7: {"Fcl_MTH_I": 0.8, "Fcl_MTH_Small": 1.0},
            8: {"Fcl_MTH_Close": 0.7, "Fcl_MTH_Neutral": 0.3},
            9: {"Fcl_MTH_U": 0.7, "Fcl_MTH_E": 0.4},
            10: {"Fcl_MTH_A": 1.0, "Fcl_MTH_Large": 0.4},
            11: {"Fcl_MTH_E": 1.0, "Fcl_MTH_Small": 0.3},
            12: {"Fcl_MTH_I": 1.0, "Fcl_MTH_Small": 0.6},
            13: {"Fcl_MTH_O": 1.0, "Fcl_MTH_U": 0.4},
            14: {"Fcl_MTH_U": 1.0, "Fcl_MTH_O": 0.4}
        }

        # Initialize OVR Lipsync context
        self.sample_rate = 44100  # OVR supports various sample rates
        self.context = None
        self.frame_delay = 0
        self.initialized = False

        logger.info("🔧 Attempting to initialize OVR Lipsync context")
        self._initialize_ovr_context()

    def _initialize_ovr_context(self):
        """Initialize OVR Lipsync context with proper error handling"""
        try:
            logger.info("🔧 Creating OVR Lipsync context")

            # Initialize OVR Lipsync
            result = ovr_lipsync.ovrLipsync_Initialize(self.sample_rate, 512)  # 512 samples buffer

            if result != ovr_lipsync.ovrLipsyncSuccess:
                logger.error(f"❌ Failed to initialize OVR Lipsync: {result}")
                raise RuntimeError(f"OVR Lipsync initialization failed with code: {result}")

            logger.info("✅ OVR Lipsync initialized successfully")

            # Create context
            self.context = ovr_lipsync.ovrLipsync_CreateContext(
                ovr_lipsync.ovrLipsyncContextProvider_Enhanced,
                self.sample_rate
            )

            if self.context is None:
                logger.error("❌ Failed to create OVR Lipsync context")
                raise RuntimeError("Failed to create OVR Lipsync context")

            logger.info(f"✅ OVR Lipsync context created: {self.context}")
            self.initialized = True

        except Exception as e:
            logger.error(f"❌ OVR Lipsync initialization failed: {e}")
            self.initialized = False
            raise

    def __del__(self):
        """Cleanup OVR Lipsync context"""
        if hasattr(self, 'context') and self.context:
            try:
                logger.info("🧹 Cleaning up OVR Lipsync context")
                ovr_lipsync.ovrLipsync_DestroyContext(self.context)
                ovr_lipsync.ovrLipsync_Shutdown()
                logger.info("✅ OVR Lipsync cleaned up successfully")
            except Exception as e:
                logger.error(f"❌ Error cleaning up OVR Lipsync: {e}")

    def convert_audio_format(self, audio_chunk, original_sample_rate: int) -> np.ndarray:
        """Convert audio to format expected by OVR Lipsync"""
        try:
            logger.debug(f"🔄 Converting audio: type={type(audio_chunk)}")

            # Handle tuple input from TTS
            if isinstance(audio_chunk, tuple):
                original_sample_rate, audio_data = audio_chunk
                audio_chunk = audio_data
                logger.debug(f"📦 Unpacked tuple: rate={original_sample_rate}")

            # Ensure numpy array
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk)
                logger.debug("🔄 Converted to numpy array")

            logger.debug(f"📊 Audio shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")

            # Convert to float32 if needed
            if audio_chunk.dtype != np.float32:
                if audio_chunk.dtype == np.int16:
                    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                elif audio_chunk.dtype == np.int32:
                    audio_chunk = audio_chunk.astype(np.float32) / 2147483648.0
                else:
                    audio_chunk = audio_chunk.astype(np.float32)
                logger.debug(f"🔄 Converted to float32")

            # Ensure mono
            if len(audio_chunk.shape) > 1:
                if audio_chunk.shape[1] > 1:
                    audio_chunk = np.mean(audio_chunk, axis=1)
                    logger.debug("🔄 Converted to mono")
                else:
                    audio_chunk = audio_chunk.flatten()

            # Resample if needed
            if original_sample_rate != self.sample_rate:
                logger.debug(f"🔄 Resampling from {original_sample_rate}Hz to {self.sample_rate}Hz")
                try:
                    import librosa
                    audio_chunk = librosa.resample(
                        audio_chunk,
                        orig_sr=original_sample_rate,
                        target_sr=self.sample_rate
                    )
                    logger.debug("✅ Resampling successful")
                except ImportError:
                    logger.warning("⚠️ librosa not available, using simple resampling")
                    # Simple resampling (not ideal but functional)
                    ratio = self.sample_rate / original_sample_rate
                    new_length = int(len(audio_chunk) * ratio)
                    audio_chunk = np.interp(
                        np.linspace(0, len(audio_chunk) - 1, new_length),
                        np.arange(len(audio_chunk)),
                        audio_chunk
                    )

            # Normalize
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = audio_chunk / max_val
                logger.debug(f"🔄 Normalized (max was {max_val:.3f})")

            logger.debug(f"✅ Audio conversion complete: {len(audio_chunk)} samples")
            return audio_chunk

        except Exception as e:
            logger.error(f"❌ Audio conversion failed: {e}")
            return np.zeros(1024, dtype=np.float32)

    def extract_visemes_from_audio(self, audio_chunk, original_sample_rate: int) -> List[OVRViseme]:
        """Extract visemes using real OVR Lipsync library"""

        if not self.initialized:
            logger.error("❌ OVR Lipsync not initialized")
            return [OVRViseme(0, 1.0, 0.0, 0.1)]

        try:
            start_time = time.time()
            logger.debug(f"🎵 Starting viseme extraction")

            # Convert audio to proper format
            processed_audio = self.convert_audio_format(audio_chunk, original_sample_rate)

            if len(processed_audio) == 0:
                logger.warning("❌ No audio data to process")
                return [OVRViseme(0, 1.0, 0.0, 0.1)]

            logger.debug(f"🎵 Processing {len(processed_audio)} samples")

            # Process audio through OVR Lipsync in chunks
            visemes = []
            chunk_size = 512  # OVR typically processes in small chunks
            total_duration = len(processed_audio) / self.sample_rate

            logger.debug(f"🔢 Processing {len(processed_audio)} samples in chunks of {chunk_size}")

            for i in range(0, len(processed_audio), chunk_size):
                chunk = processed_audio[i:i + chunk_size]

                # Pad chunk if needed
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                chunk_start_time = i / self.sample_rate
                chunk_end_time = min((i + chunk_size) / self.sample_rate, total_duration)

                logger.debug(f"🎵 Processing chunk {i//chunk_size + 1}: {len(chunk)} samples, time: {chunk_start_time:.3f}s-{chunk_end_time:.3f}s")

                # Call OVR Lipsync
                try:
                    # Process frame with OVR Lipsync
                    result = ovr_lipsync.ovrLipsync_ProcessFrame(
                        self.context,
                        chunk,
                        len(chunk),
                        self.frame_delay
                    )

                    if result != ovr_lipsync.ovrLipsyncSuccess:
                        logger.warning(f"⚠️ OVR ProcessFrame returned: {result}")
                        continue

                    # Get viseme frame
                    viseme_frame = ovr_lipsync.ovrLipsync_GetVisemeFrame(self.context)

                    if viseme_frame is None:
                        logger.warning("⚠️ No viseme frame returned")
                        continue

                    logger.debug(f"🎭 Got viseme frame with {len(viseme_frame.visemes)} visemes")

                    # Convert OVR visemes to our format
                    for viseme_id, weight in enumerate(viseme_frame.visemes):
                        if weight > 0.01:  # Only include visemes with meaningful weight
                            viseme = OVRViseme(
                                viseme_id=viseme_id,
                                weight=weight,
                                start_time=chunk_start_time,
                                end_time=chunk_end_time
                            )
                            visemes.append(viseme)
                            logger.debug(f"🎭 Viseme {viseme_id} ({self.ovr_viseme_names.get(viseme_id, 'unknown')}): weight={weight:.3f}")

                except Exception as e:
                    logger.error(f"❌ OVR processing error for chunk {i//chunk_size}: {e}")
                    continue

            processing_time = time.time() - start_time
            logger.info(f"✅ Extracted {len(visemes)} visemes in {processing_time:.3f}s")

            # Return default silence if no visemes found
            if not visemes:
                logger.warning("⚠️ No visemes extracted, returning silence")
                return [OVRViseme(0, 1.0, 0.0, total_duration)]

            return visemes

        except Exception as e:
            logger.error(f"❌ Viseme extraction failed: {e}")
            return [OVRViseme(0, 1.0, 0.0, 0.1)]

    def visemes_to_blend_shapes(self, visemes: List[OVRViseme]) -> Dict[str, float]:
        """Convert OVR visemes to VRM blend shapes"""
        try:
            logger.debug(f"🎨 Converting {len(visemes)} visemes to blend shapes")

            # Initialize blend shapes
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

            # Accumulate blend shapes from all visemes
            total_weight = 0.0
            for viseme in visemes:
                if viseme.viseme_id in self.ovr_to_vrm_mapping:
                    target_shapes = self.ovr_to_vrm_mapping[viseme.viseme_id]

                    logger.debug(f"🎭 Processing viseme {viseme.viseme_id} ({self.ovr_viseme_names.get(viseme.viseme_id, 'unknown')}) weight={viseme.weight:.3f}")

                    for shape_name, base_weight in target_shapes.items():
                        if shape_name in blend_shapes:
                            contribution = base_weight * viseme.weight
                            blend_shapes[shape_name] += contribution
                            logger.debug(f"  📊 {shape_name} += {contribution:.3f}")

                    total_weight += viseme.weight

            # Normalize if we have accumulated weights
            if total_weight > 0:
                for shape_name in blend_shapes:
                    blend_shapes[shape_name] = min(1.0, blend_shapes[shape_name])
                logger.debug(f"🎨 Normalized blend shapes (total weight: {total_weight:.3f})")
            else:
                blend_shapes["Fcl_MTH_Neutral"] = 1.0

            # Log final active shapes
            active_shapes = {k: v for k, v in blend_shapes.items() if v > 0.01}
            logger.debug(f"✅ Final active blend shapes: {active_shapes}")

            return blend_shapes

        except Exception as e:
            logger.error(f"❌ Blend shape conversion failed: {e}")
            return {"Fcl_MTH_Neutral": 1.0}

    def process_audio_chunk(self, audio_chunk, original_sample_rate: int) -> Dict[str, float]:
        """Complete pipeline: audio -> visemes -> blend shapes"""
        try:
            logger.debug(f"🔄 Processing audio chunk pipeline")

            # Extract visemes
            visemes = self.extract_visemes_from_audio(audio_chunk, original_sample_rate)

            # Convert to blend shapes
            blend_shapes = self.visemes_to_blend_shapes(visemes)

            logger.debug(f"✅ Audio processing complete")
            return blend_shapes

        except Exception as e:
            logger.error(f"❌ Audio processing pipeline failed: {e}")
            return {"Fcl_MTH_Neutral": 1.0}