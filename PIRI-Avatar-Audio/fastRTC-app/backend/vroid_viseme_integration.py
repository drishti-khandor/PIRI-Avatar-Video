"""
VRoid Viseme Integration for Unified Server
Integrates the advanced VRoid viseme system with your existing backend
Replace the existing VisemeController in unified_server.py with this enhanced version
"""

import asyncio
import json
import os
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from viseme_extractor import VisemeExtractor, AdvancedVisemeExtractor
from dotenv import load_dotenv
from fastapi import WebSocket
import numpy as np
from fastrtc import AdditionalOutputs, get_stt_model, get_tts_model
from openai import AzureOpenAI


# Import the advanced viseme mapper (place this in the same directory as unified_server.py)
from advanced_vroid_viseme_system import AdvancedVRoidVisemeMapper, VRoidVisemeAnimator, VisemeTransitionType
load_dotenv()
logger = logging.getLogger(__name__)
# Environment setup for Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# System prompt
sys_prompt = """You are a helpful AI assistant with a 3D avatar. Keep responses concise and natural for speech synthesis."""
messages = [{"role": "system", "content": sys_prompt}]

# Initialize AI models
if not all([azure_endpoint, api_key, deployment_name]):
    logger.warning("Missing Azure OpenAI environment variables. AI features will be limited.")
    openai_client = None
else:
    openai_client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )
stt_model = get_stt_model()
tts_model = get_tts_model(model="kokoro")


# Initialize viseme extractor
try:
    viseme_extractor = AdvancedVisemeExtractor()
    logger.info("Using advanced viseme extractor")
except:
    viseme_extractor = VisemeExtractor()
    logger.info("Using basic viseme extractor")
@dataclass
class EnhancedVisemeData:
    """Enhanced viseme data with additional context"""
    viseme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    phoneme: str = ""
    emotion: str = "neutral"


class EnhancedVRoidVisemeController:
    """
    Enhanced VRoid Viseme Controller with advanced blend shape mapping
    Replaces the basic VisemeController in unified_server.py
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

        # Initialize advanced viseme mapping system
        self.viseme_mapper = AdvancedVRoidVisemeMapper()
        self.animator = VRoidVisemeAnimator(self.viseme_mapper)

        # Current state
        self.current_blend_shapes = {}
        self.current_emotion = "neutral"
        self.animation_queue = []
        self.is_animating = False

        # Threading for smooth animation
        self.animation_lock = threading.Lock()
        self.should_stop_animation = False

        # Start the animation system
        self.animator.start_animation_loop()

        logger.info("✅ Enhanced VRoid Viseme Controller initialized")

    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ Avatar client connected. Total: {len(self.active_connections)}")

        # Send initial neutral state
        await self._broadcast_blend_shapes(self.viseme_mapper.get_instantaneous_viseme_weights('sil'))

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"❌ Avatar client disconnected. Total: {len(self.active_connections)}")

    async def _broadcast_blend_shapes(self, blend_shapes: Dict[str, float]):
        """Broadcast blend shape updates to all connected clients"""
        if not self.active_connections:
            logger.debug("❌ No avatar connections to broadcast to")
            return

        # Update current state
        self.current_blend_shapes = blend_shapes.copy()

        # Create message with VRoid-specific format
        message = {
            "type": "viseme_update",
            "blend_shapes": blend_shapes,
            "timestamp": time.time(),
            "emotion": self.current_emotion
        }

        logger.debug(f"📡 Broadcasting to {len(self.active_connections)} connections: {len(blend_shapes)} blend shapes")

        # Send to all connections
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send to connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def update_from_ai_visemes(self, ai_visemes: List[Dict], emotion: str = "neutral"):
        """
        Process AI-generated visemes using advanced VRoid mapping
        This replaces the simple mapping in the original code
        """
        if not ai_visemes:
            # Return to neutral/rest position
            neutral_weights = self.viseme_mapper.get_instantaneous_viseme_weights('sil', emotion)
            await self._broadcast_blend_shapes(neutral_weights)
            return

        logger.info(f"🎭 Processing {len(ai_visemes)} AI visemes with emotion: {emotion}")

        # Convert AI visemes to phoneme sequence
        phoneme_sequence = []
        for viseme_data in ai_visemes:
            # Extract data from AI viseme
            ai_viseme_id = str(viseme_data.get('viseme', '0'))
            start_time = float(viseme_data.get('start_time', 0.0))
            end_time = float(viseme_data.get('end_time', 0.1))
            confidence = float(viseme_data.get('confidence', 1.0))

            # Map AI viseme to phoneme using improved mapping
            phoneme = self._ai_viseme_to_phoneme(ai_viseme_id)
            phoneme_sequence.append((phoneme, start_time, end_time))

        # Generate smooth animation sequence
        try:
            animation_frames = self.viseme_mapper.process_phoneme_sequence(
                phoneme_sequence,
                emotion=emotion,
                transition_type=VisemeTransitionType.SMOOTH
            )

            # Start animation playback
            await self._play_animation_sequence(animation_frames)

        except Exception as e:
            logger.error(f"Failed to process viseme sequence: {e}")
            # Fallback to simple viseme
            fallback_weights = self.viseme_mapper.get_instantaneous_viseme_weights('sil', emotion)
            await self._broadcast_blend_shapes(fallback_weights)

    def _ai_viseme_to_phoneme(self, ai_viseme_id: str) -> str:
        """
        Enhanced mapping from AI viseme IDs to phonemes
        Based on the Microsoft Speech Platform viseme mapping you provided
        """
        viseme_to_phoneme_map = {
            '0': 'sil',  # Silence
            '1': 'AH',  # Open vowels (AE, AH, EH, UH)
            '2': 'AA',  # Open back vowels (AA, AO, AW, OW)
            '3': 'EY',  # Diphthongs (AY, EY, OY)
            '4': 'ER',  # R-colored vowels (ER, AX, IX)
            '5': 'IH',  # Close front vowels (IH, IY)
            '6': 'UW',  # Close back vowels (UW, UH)
            '7': 'P',  # Bilabials (B, P, M)
            '8': 'F',  # Labiodentals (F, V)
            '9': 'TH',  # Dental fricatives (TH, DH)
            '10': 'T',  # Alveolars (T, D, N, L)
            '11': 'S',  # Sibilants (S, Z)
            '12': 'SH',  # Post-alveolars (SH, ZH, CH, JH)
            '13': 'K',  # Velars (K, G, NG)
            '14': 'R',  # Approximants (R, W, Y, HH)
        }

        phoneme = viseme_to_phoneme_map.get(ai_viseme_id, 'sil')
        logger.debug(f"Mapped AI viseme '{ai_viseme_id}' to phoneme '{phoneme}'")
        return phoneme

    # async def _play_animation_sequence(self, frames: List):
    #     """Play back a sequence of animation frames in real-time"""
    #     if not frames:
    #         return
    #
    #     logger.info(f"🎬 Starting animation playback with {len(frames)} frames")
    #
    #     start_time = time.time()
    #
    #     for frame in frames:
    #         # Calculate when this frame should be displayed
    #         target_time = start_time + frame.timestamp
    #         current_time = time.time()
    #
    #         # Wait until it's time for this frame
    #         if target_time > current_time:
    #             await asyncio.sleep(target_time - current_time)
    #
    #         # Send the frame
    #         await self._broadcast_blend_shapes(frame.blend_shapes)
    #
    #     logger.info("✅ Animation playback completed")

    async def _play_animation_sequence(self, frames: List):
        """Enhanced debug version of animation playback"""
        if not frames:
            logger.warning("❌ No frames provided to _play_animation_sequence")
            return

        logger.info(f"🎬 Starting animation playback with {len(frames)} frames")

        # Debug: Log first few frames to check structure
        for i, frame in enumerate(frames[:3]):
            logger.info(f"🎭 Frame {i}: timestamp={getattr(frame, 'timestamp', 'MISSING')}, "
                        f"blend_shapes_count={len(getattr(frame, 'blend_shapes', {}))}")
            if hasattr(frame, 'blend_shapes'):
                logger.info(f"    Blend shapes sample: {dict(list(frame.blend_shapes.items())[:3])}")

        # Debug: Check WebSocket connections
        logger.info(f"📡 Active WebSocket connections: {len(self.active_connections)}")
        if not self.active_connections:
            logger.error("❌ No active WebSocket connections - lip sync will not work!")
            return

        start_time = time.time()
        frames_sent = 0
        errors = 0

        for i, frame in enumerate(frames):
            try:
                # Debug: Check frame structure
                if not hasattr(frame, 'timestamp'):
                    logger.error(f"❌ Frame {i} missing timestamp attribute")
                    continue

                if not hasattr(frame, 'blend_shapes'):
                    logger.error(f"❌ Frame {i} missing blend_shapes attribute")
                    continue

                # Calculate timing
                target_time = start_time + frame.timestamp
                current_time = time.time()
                wait_time = target_time - current_time

                # Debug: Log timing for first few frames
                if i < 5:
                    logger.info(f"⏱️ Frame {i}: target={target_time:.3f}, current={current_time:.3f}, "
                                f"wait={wait_time:.3f}s")

                # Wait until it's time for this frame (but don't wait too long)
                if wait_time > 0 and wait_time < 1.0:  # Cap wait time to 1 second
                    await asyncio.sleep(wait_time)
                elif wait_time >= 1.0:
                    logger.warning(f"⚠️ Frame {i} wait time too long: {wait_time:.3f}s, skipping wait")

                # Send the frame with debug info
                logger.debug(f"📤 Sending frame {i} with {len(frame.blend_shapes)} blend shapes")

                # Debug: Log significant blend shapes
                significant_shapes = {k: v for k, v in frame.blend_shapes.items() if v > 0.1}
                if significant_shapes:
                    logger.info(f"🎭 Frame {i} significant shapes: {significant_shapes}")

                await self._broadcast_blend_shapes(frame.blend_shapes)
                frames_sent += 1

            except Exception as e:
                errors += 1
                logger.error(f"❌ Error processing frame {i}: {e}")

            # Emergency brake - if too many errors, stop
            if errors > 10:
                logger.error("❌ Too many errors, stopping animation playback")
                break

        logger.info(f"✅ Animation playback completed: {frames_sent}/{len(frames)} frames sent, {errors} errors")

        # Final debug check
        if frames_sent == 0:
            logger.error("❌ NO FRAMES WERE SENT - This is why lip sync isn't working!")
            logger.error("🔍 Check: 1) Frame structure, 2) WebSocket connections, 3) Timing logic")

    async def _broadcast_blend_shapes(self, blend_shapes: Dict[str, float]):
        """Enhanced debug version of blend shape broadcasting"""
        if not self.active_connections:
            logger.error("❌ No active WebSocket connections for broadcasting")
            return

        # Debug: Validate blend shapes
        if not blend_shapes:
            logger.warning("⚠️ Empty blend_shapes dict")
            return

        # Debug: Check for significant values
        significant_shapes = {k: v for k, v in blend_shapes.items() if v > 0.01}
        if not significant_shapes:
            logger.warning("⚠️ No significant blend shape values (all < 0.01)")
        else:
            logger.debug(f"📊 Broadcasting {len(significant_shapes)} significant blend shapes")

        # Update current state
        self.current_blend_shapes = blend_shapes.copy()

        # Create message with debug info
        message = {
            "type": "viseme_update",
            "blend_shapes": blend_shapes,
            "timestamp": time.time(),
            "emotion": self.current_emotion,
            "debug_info": {
                "significant_shapes_count": len(significant_shapes),
                "total_shapes_count": len(blend_shapes),
                "max_weight": max(blend_shapes.values()) if blend_shapes else 0
            }
        }

        logger.debug(f"📡 Broadcasting to {len(self.active_connections)} connections")

        # Send to all connections with error handling
        disconnected = []
        sent_count = 0

        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
                sent_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to send to WebSocket connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

        logger.debug(f"📤 Sent to {sent_count}/{len(self.active_connections)} connections")

        if sent_count == 0:
            logger.error("❌ FAILED TO SEND TO ANY WEBSOCKET CONNECTIONS!")

    # Also add this debug function to test WebSocket connectivity
    async def test_websocket_broadcast(self):
        """Test function to verify WebSocket broadcasting works"""
        logger.info("🧪 Testing WebSocket broadcast...")

        test_blend_shapes = {
            "Fcl_MTH_A": 0.8,
            "Fcl_MTH_Close": 0.2,
            "Fcl_EYE_Natural": 0.9
        }

        await self._broadcast_blend_shapes(test_blend_shapes)
        logger.info("🧪 Test broadcast completed")

    async def update_single_viseme(self, phoneme: str, emotion: str = "neutral"):
        """Update to a single viseme immediately (for manual control)"""
        weights = self.viseme_mapper.get_instantaneous_viseme_weights(phoneme, emotion)
        await self._broadcast_blend_shapes(weights)
        logger.debug(f"🎯 Updated to single viseme: {phoneme} with emotion: {emotion}")

    async def set_emotion(self, emotion: str):
        """Set the current emotional context"""
        if emotion in ['neutral', 'happy', 'sad', 'surprised', 'angry']:
            self.current_emotion = emotion
            # Update current viseme with new emotion
            if hasattr(self, '_last_phoneme'):
                await self.update_single_viseme(self._last_phoneme, emotion)
            logger.info(f"😊 Emotion set to: {emotion}")
        else:
            logger.warning(f"Unknown emotion: {emotion}")

    async def play_phoneme_sequence(self, phoneme_sequence: List[Tuple[str, float, float]], emotion: str = "neutral"):
        """
        Play a sequence of phonemes with timing
        This is for the /play_phonemes endpoint
        """
        try:
            animation_frames = self.viseme_mapper.process_phoneme_sequence(
                phoneme_sequence,
                emotion=emotion,
                transition_type=VisemeTransitionType.SMOOTH
            )

            await self._play_animation_sequence(animation_frames)

        except Exception as e:
            logger.error(f"Failed to play phoneme sequence: {e}")

    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions"""
        return ['neutral', 'happy', 'sad', 'surprised', 'angry']

    def get_current_state(self) -> Dict:
        """Get current controller state for debugging"""
        return {
            'connected_clients': len(self.active_connections),
            'current_emotion': self.current_emotion,
            'current_blend_shapes': self.current_blend_shapes,
            'is_animating': self.is_animating
        }

    async def reset_to_neutral(self):
        """Reset avatar to neutral expression"""
        neutral_weights = self.viseme_mapper.get_instantaneous_viseme_weights('sil', 'neutral')
        await self._broadcast_blend_shapes(neutral_weights)
        self.current_emotion = 'neutral'
        logger.info("🔄 Reset to neutral expression")


# Function to replace the existing process_audio_and_respond function
def enhanced_process_audio_and_respond(audio, enhanced_viseme_controller: EnhancedVRoidVisemeController):
    """
    Enhanced audio processing function with advanced VRoid viseme integration
    Replace the existing process_audio_and_respond function with this
    """
    # ... (STT and LLM code remains the same until TTS section) ...

    # Speech-to-Text (same as before)
    stt_time = time.time()
    logger.info("Performing STT")
    text = stt_model.stt(audio)
    if not text:
        logger.info("STT returned empty string")
        return

    logger.info(f"STT response: {text}")
    yield AdditionalOutputs({"type": "stt", "text": text})

    messages.append({"role": "user", "content": text})
    logger.info(f"STT took {time.time() - stt_time} seconds")

    # LLM Generation (same as before)
    llm_time = time.time()
    try:
        if openai_client:
            response = openai_client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            full_response = response.choices[0].message.content
        else:
            full_response = "AI service not configured. Please check your environment variables."
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        full_response = "I'm having trouble processing that right now."

    logger.info(f"LLM response: {full_response}")
    logger.info(f"LLM took {time.time() - llm_time} seconds")
    yield AdditionalOutputs({"type": "llm", "text": full_response})

    # ENHANCED TTS with Advanced VRoid Viseme Integration
    logger.info("Starting enhanced TTS streaming with advanced VRoid visemes.")
    chunk_index = 0
    accumulated_time = 0.0
    all_visemes = []

    try:
        for sample_rate, audio_chunk in tts_model.stream_tts_sync(full_response):
            # Calculate timing for this chunk
            chunk_duration = len(audio_chunk) / sample_rate
            chunk_start_time = accumulated_time
            accumulated_time += chunk_duration

            # Extract visemes from this audio chunk
            try:
                visemes = viseme_extractor.extract_visemes_from_chunk(audio_chunk, sample_rate)

                # Adjust viseme timing to global timeline
                enhanced_visemes = []
                for viseme in visemes:
                    enhanced_viseme = {
                        "viseme": str(viseme.viseme),
                        "start_time": float(viseme.start_time + chunk_start_time),
                        "end_time": float(viseme.end_time + chunk_start_time),
                        "confidence": float(viseme.confidence)
                    }
                    enhanced_visemes.append(enhanced_viseme)
                    all_visemes.append(enhanced_viseme)

                # Send viseme data to frontend
                viseme_data = {
                    "type": "visemes",
                    "chunk_index": chunk_index,
                    "visemes": enhanced_visemes,
                    "chunk_duration": float(chunk_duration),
                    "chunk_start_time": float(chunk_start_time)
                }

                logger.info(f"Enhanced visemes for chunk {chunk_index}: {[v['viseme'] for v in enhanced_visemes]}")
                yield AdditionalOutputs(viseme_data)

                # ENHANCED: Update avatar with advanced VRoid system
                def update_avatar_enhanced():
                    try:
                        # Detect emotion from text (simple keyword-based)
                        emotion = detect_emotion_from_text(full_response)

                        # Create new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        # Update avatar with enhanced visemes
                        loop.run_until_complete(
                            enhanced_viseme_controller.update_from_ai_visemes(enhanced_visemes, emotion)
                        )

                        loop.close()

                    except Exception as e:
                        logger.error(f"Failed to update enhanced avatar: {e}")

                # Start avatar update in background thread
                threading.Thread(target=update_avatar_enhanced, daemon=True).start()

            except Exception as e:
                logger.error(f"Viseme extraction failed for chunk {chunk_index}: {e}")
                # Send default silence viseme
                yield AdditionalOutputs({
                    "type": "visemes",
                    "chunk_index": chunk_index,
                    "visemes": [{"viseme": "0", "start_time": chunk_start_time, "end_time": accumulated_time,
                                 "confidence": 1.0}],
                    "chunk_duration": chunk_duration,
                    "chunk_start_time": chunk_start_time
                })

            # Yield the audio chunk for playback
            yield sample_rate, audio_chunk
            chunk_index += 1

        logger.info("Finished enhanced TTS streaming with advanced visemes.")

    except Exception as e:
        logger.error(f"Enhanced TTS failed: {e}")

    messages.append({"role": "assistant", "content": full_response + " "})


def detect_emotion_from_text(text: str) -> str:
    """
    Simple emotion detection from text
    Replace with more sophisticated sentiment analysis if needed
    """
    text_lower = text.lower()

    # Happy indicators
    if any(word in text_lower for word in
           ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', '!', 'haha', 'lol']):
        return 'happy'

    # Sad indicators
    if any(word in text_lower for word in ['sad', 'sorry', 'unfortunately', 'terrible', 'awful', 'disappointed']):
        return 'sad'

    # Surprised indicators
    if any(word in text_lower for word in ['wow', 'amazing', 'incredible', 'unbelievable', 'really?', 'no way', '!']):
        return 'surprised'

    # Angry indicators
    if any(word in text_lower for word in ['angry', 'mad', 'frustrated', 'annoying', 'ridiculous', 'stupid']):
        return 'angry'

    return 'neutral'


# Additional endpoints for enhanced viseme control
async def set_avatar_emotion_endpoint(emotion: str, enhanced_controller: EnhancedVRoidVisemeController):
    """Endpoint to manually set avatar emotion"""
    await enhanced_controller.set_emotion(emotion)
    return {"status": "success", "emotion": emotion}


async def trigger_manual_viseme_endpoint(phoneme: str, emotion: str,
                                         enhanced_controller: EnhancedVRoidVisemeController):
    """Endpoint to manually trigger a viseme"""
    await enhanced_controller.update_single_viseme(phoneme, emotion)
    return {"status": "success", "phoneme": phoneme, "emotion": emotion}


async def reset_avatar_endpoint(enhanced_controller: EnhancedVRoidVisemeController):
    """Endpoint to reset avatar to neutral"""
    await enhanced_controller.reset_to_neutral()
    return {"status": "success", "message": "Avatar reset to neutral"}


def get_avatar_status_endpoint(enhanced_controller: EnhancedVRoidVisemeController):
    """Endpoint to get current avatar status"""
    return enhanced_controller.get_current_state()