"""
Audio chat processor that integrates audio, AI, and viseme processing
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, AsyncGenerator, Dict, Any
import base64

from fastrtc.utils import audio_to_bytes
from fastrtc import get_stt_model, get_tts_model, AdditionalOutputs

from app.core.avatar.viseme_controller import VisemeAnimationController
from app.core.audio.audio_processor import AudioProcessor
from app.core.ai.llm_client import LLMClient
from app.models.avatar import VisemeData, EmotionType


logger = logging.getLogger(__name__)


class AudioChatProcessor:
    """
    Processes audio input through the complete pipeline:
    Audio -> STT -> LLM -> TTS -> Visemes -> Avatar Animation
    """

    def __init__(self, viseme_controller: VisemeAnimationController):
        self.viseme_controller = viseme_controller
        self.audio_processor = AudioProcessor()
        self.llm_client = LLMClient()

        # Initialize STT and TTS models
        self.stt_model = get_stt_model()
        self.tts_model = get_tts_model(model="kokoro")

        logger.info("AudioChatProcessor initialized")

    def process_audio_and_respond(
        self, audio
    ):
        """
        Process audio input and generate synchronized response
        Note: This is NOT async - it's a generator function called by fastrtc
        """
        try:
            # Debug log the incoming data type
            logger.debug(f"Incoming audio type: {type(audio)}, value preview: {str(audio)[:100] if not isinstance(audio, np.ndarray) else 'numpy array'}")
            
            # Handle audio input - it might be a tuple (sample_rate, audio_data)
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                logger.info(f"Received audio tuple: sample_rate={sample_rate}")
                audio = audio_data
            
            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Log incoming audio data
            logger.info(f"Received audio data: shape={audio.shape}, dtype={audio.dtype}, duration={len(audio)/sample_rate:.2f}s")
            logger.debug(f"Audio stats: min={audio.min()}, max={audio.max()}, mean={audio.mean():.4f}, std={audio.std():.4f}")
            
            # 1. Speech-to-Text
            logger.info("Starting STT processing...")
            # STT expects a tuple of (sample_rate, audio_data)
            stt_result = self.stt_model.stt((sample_rate, audio))

            if not stt_result or not stt_result.strip():
                logger.info("No speech detected")
                return

            logger.info(f"STT Result: {stt_result}")

            # Yield STT result
            yield AdditionalOutputs({
                "type": "stt",
                "text": stt_result,
                "timestamp": time.time()
            })

            # 2. Generate AI Response (run in thread to avoid blocking)
            logger.info("Generating AI response...")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.llm_client.generate_response_sync, stt_result)
                ai_response, emotion = future.result()

            logger.info(f"AI Response: {ai_response} (Emotion: {emotion})")

            # Yield LLM response
            yield AdditionalOutputs({
                "type": "llm",
                "text": ai_response,
                "emotion": emotion.value,
                "timestamp": time.time()
            })

            # 3. Text-to-Speech with viseme extraction
            logger.info("Starting TTS streaming with viseme extraction.")
            chunk_index = 0
            accumulated_time = 0.0
            all_audio_chunks = []
            all_visemes = []

            try:
                # First, collect all audio chunks
                tts_sample_rate = None
                
                for sample_rate, audio_chunk in self.tts_model.stream_tts_sync(ai_response):
                    if tts_sample_rate is None:
                        tts_sample_rate = sample_rate
                    
                    # Calculate timing for this chunk
                    chunk_duration = len(audio_chunk) / sample_rate
                    chunk_start_time = accumulated_time
                    accumulated_time += chunk_duration

                    all_audio_chunks.append(audio_chunk)
                    
                    # Log chunk info
                    logger.debug(f"TTS chunk {chunk_index}: duration={chunk_duration:.3f}s, sample_rate={sample_rate}")

                    # For now, send placeholder visemes during streaming
                    yield AdditionalOutputs({
                        "type": "visemes",
                        "chunk_index": chunk_index,
                        "visemes": [{
                            "viseme": "sil",
                            "start_time": chunk_start_time,
                            "end_time": chunk_start_time + chunk_duration,
                            "confidence": 0.5,
                            "emotion": emotion.value
                        }],
                        "chunk_duration": chunk_duration,
                        "chunk_start_time": chunk_start_time
                    })

                    # Yield the audio chunk for playback
                    yield sample_rate, audio_chunk
                    chunk_index += 1

                logger.info("Finished TTS streaming. Now extracting visemes from complete audio.")

                # Combine all audio chunks and extract visemes from the complete audio
                if all_audio_chunks and tts_sample_rate:
                    combined_audio = np.concatenate(all_audio_chunks)
                    
                    # Extract visemes from the complete audio
                    try:
                        logger.info(f"Extracting visemes from complete audio: duration={len(combined_audio)/tts_sample_rate:.3f}s")
                        all_visemes = self.audio_processor.extract_visemes_sync(
                            combined_audio, ai_response, tts_sample_rate
                        )
                        
                        # Add emotion to all visemes
                        for viseme in all_visemes:
                            viseme.emotion = emotion
                            
                        logger.info(f"Successfully extracted {len(all_visemes)} visemes from complete audio")
                        
                    except Exception as e:
                        logger.error(f"Failed to extract visemes from complete audio: {e}")
                        # Generate fallback visemes
                        all_visemes = self.audio_processor.generate_fallback_visemes(
                            ai_response, len(combined_audio) / tts_sample_rate
                        )
                        for viseme in all_visemes:
                            viseme.emotion = emotion
                    
                    audio_bytes = audio_to_bytes((tts_sample_rate, combined_audio))
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # Final response with all data
                    yield AdditionalOutputs({
                        "type": "tts_response",
                        "text": ai_response,
                        "audio_b64": audio_b64,
                        "visemes": [v.dict() for v in all_visemes],
                        "emotion": emotion.value,
                        "audio_duration": accumulated_time,
                        "timestamp": time.time()
                    })

            except Exception as e:
                logger.error(f"TTS failed: {e}")
                yield AdditionalOutputs({
                    "type": "error",
                    "message": str(e),
                    "timestamp": time.time()
                })

        except Exception as e:
            logger.error(f"Error in unified processing: {e}", exc_info=True)
            yield AdditionalOutputs({
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            })
