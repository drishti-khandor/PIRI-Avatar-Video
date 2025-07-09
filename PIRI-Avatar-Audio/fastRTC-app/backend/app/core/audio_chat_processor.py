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
from fastrtc import get_stt_model, get_tts_model

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
    
    async def process_audio_and_respond(
        self, 
        audio: np.ndarray
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process audio input and generate synchronized response
        """
        try:
            # 1. Speech-to-Text
            logger.info("Starting STT processing...")
            stt_result = await asyncio.to_thread(
                self.stt_model, audio
            )
            
            if not stt_result or not stt_result.strip():
                logger.info("No speech detected")
                return
            
            logger.info(f"STT Result: {stt_result}")
            
            # Yield STT result
            yield {
                "type": "stt",
                "text": stt_result,
                "timestamp": time.time()
            }
            
            # 2. Generate AI Response
            logger.info("Generating AI response...")
            ai_response, emotion = await self.llm_client.generate_response(stt_result)
            
            logger.info(f"AI Response: {ai_response} (Emotion: {emotion})")
            
            # 3. Text-to-Speech
            logger.info("Starting TTS processing...")
            tts_audio = await asyncio.to_thread(
                self.tts_model, ai_response
            )
            
            if tts_audio is None:
                logger.error("TTS failed to generate audio")
                return
            
            # Convert audio to proper format
            audio_bytes = audio_to_bytes(tts_audio)
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Calculate audio duration
            sample_rate = 16000  # Default sample rate
            audio_duration = len(tts_audio) / sample_rate
            
            # 4. Extract visemes
            logger.info("Extracting visemes...")
            visemes = await self.audio_processor.extract_visemes(
                tts_audio, 
                ai_response,
                sample_rate
            )
            
            # If ForceAlign fails, use fallback
            if not visemes:
                logger.warning("ForceAlign failed, using fallback visemes")
                visemes = self.audio_processor.generate_fallback_visemes(
                    ai_response,
                    audio_duration
                )
            
            # Add emotion to all visemes
            for viseme in visemes:
                viseme.emotion = emotion
            
            # 5. Schedule viseme animation
            logger.info(f"Scheduling {len(visemes)} visemes for animation")
            await self.viseme_controller.add_viseme_sequence(visemes)
            
            # 6. Yield complete response
            yield {
                "type": "tts_response",
                "text": ai_response,
                "audio_b64": audio_b64,
                "visemes": [v.dict() for v in visemes],
                "emotion": emotion.value,
                "audio_duration": audio_duration,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in unified processing: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            }
