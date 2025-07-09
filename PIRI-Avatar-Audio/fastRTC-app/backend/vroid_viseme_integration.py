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
from ForceAlign_viseme_integration import ForceAlignVisemeExtractor
from dotenv import load_dotenv
from fastapi import WebSocket
import numpy as np
from fastrtc import AdditionalOutputs, get_stt_model, get_tts_model
from openai import AzureOpenAI


# Import the simplified VRoid viseme mapper
from vroid_viseme_mapper import VRoidVisemeMapper
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


# Initialize ForceAlign viseme extractor
try:
    viseme_extractor = ForceAlignVisemeExtractor()
    logger.info("Using ForceAlign viseme extractor")
except Exception as e:
    logger.error(f"Failed to initialize ForceAlign viseme extractor: {e}")
    viseme_extractor = None
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

        # Initialize simplified viseme mapping system
        self.viseme_mapper = VRoidVisemeMapper()

        # Current state
        self.current_blend_shapes = {}
        self.animation_queue = []
        self.is_animating = False
        self.animation_lock = asyncio.Lock()

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
            "timestamp": time.time()
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

    async def update_from_ai_visemes(self, ai_visemes: List[Dict]):
        """
        Process ForceAlign-generated visemes and convert to VRoid blend shapes
        """
        if not ai_visemes:
            # Return to neutral/rest position
            neutral_weights = self.viseme_mapper.get_instantaneous_viseme_weights('sil')
            await self._broadcast_blend_shapes(neutral_weights)
            logger.info("🔄 No AI visemes - returning to neutral/rest position")
            return

        logger.info(f"🎭 Processing {len(ai_visemes)} ForceAlign visemes")

        # Process each viseme with proper timing
        for i, viseme_data in enumerate(ai_visemes):
            try:
                # Extract data from ForceAlign viseme
                viseme_string = str(viseme_data.get('viseme', 'sil'))
                start_time = float(viseme_data.get('start_time', 0.0))
                end_time = float(viseme_data.get('end_time', 0.1))
                confidence = float(viseme_data.get('confidence', 1.0))

                logger.info(
                    f"🔢 VISEME[{i}]: {viseme_string}, time={start_time:.3f}-{end_time:.3f}, confidence={confidence}")

                # Get blend shape weights for this viseme
                blend_shapes = self.viseme_mapper.get_instantaneous_viseme_weights(viseme_string)
                
                # Log the blend shapes being applied
                active_shapes = {k: v for k, v in blend_shapes.items() if v > 0.01}
                logger.info(f"🎯 BLEND_SHAPES[{i}]: {active_shapes}")

                # Apply the blend shapes
                await self._broadcast_blend_shapes(blend_shapes)
                
                # Calculate hold duration based on viseme duration
                duration = end_time - start_time
                hold_time = max(0.1, min(duration, 0.5))  # Between 100ms and 500ms
                
                # Hold the viseme for the calculated duration
                await asyncio.sleep(hold_time)
                
            except Exception as e:
                logger.error(f"Failed to process viseme {i}: {e}")
                continue

        # Return to neutral after all visemes
        neutral_weights = self.viseme_mapper.get_instantaneous_viseme_weights('sil')
        await self._broadcast_blend_shapes(neutral_weights)
        logger.info("✅ All visemes processed, returned to neutral")


    async def _play_animation_sequence(self, frames: List):
        """DISABLED - Use direct updates instead"""
        logger.info("⚠️ _play_animation_sequence is disabled - using direct updates")
        return

        # STOP ANY EXISTING ANIMATION FIRST
        self.is_animating = False
        await asyncio.sleep(0.01)  # Brief pause to let current animation stop
        self.is_animating = True

        logger.info(f"🎬 Starting animation playback with {len(frames)} frames")

        # Debug: Log first few frames to check structure
        for i, frame in enumerate(frames[:3]):
            if not self.is_animating:
                logger.info(f"🛑 Animation stopped at frame {i}")
                break
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
        # ADD THESE LINES:
        self.is_animating = False  # Mark animation as complete
        logger.info("🏁 Animation state reset")
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

    async def update_single_viseme(self, viseme: str):
        """Update to a single viseme immediately (for manual control)"""
        weights = self.viseme_mapper.get_instantaneous_viseme_weights(viseme)
        await self._broadcast_blend_shapes(weights)
        # ADD LOGGING HERE:
        logger.info(f"🎯 VISEME: {viseme} -> weights: {list(weights.keys())}")
        logger.debug(f"🎯 Updated to single viseme: {viseme}")


    def get_current_state(self) -> Dict:
        """Get current controller state for debugging"""
        return {
            'connected_clients': len(self.active_connections),
            'current_blend_shapes': self.current_blend_shapes,
            'is_animating': self.is_animating
        }

    async def reset_to_neutral(self):
        """Reset avatar to neutral expression"""
        neutral_weights = self.viseme_mapper.get_instantaneous_viseme_weights('sil')
        await self._broadcast_blend_shapes(neutral_weights)
        logger.info("🔄 Reset to neutral expression")


# Function to replace the existing process_audio_and_respond function
async def enhanced_process_audio_and_respond(audio, enhanced_viseme_controller: EnhancedVRoidVisemeController):
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

    # ENHANCED TTS with ForceAlign Viseme Integration
    logger.info("Starting enhanced TTS streaming with ForceAlign visemes.")
    
    # Collect all audio chunks first
    audio_chunks = []
    sample_rate = None
    
    tts_data = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: list(tts_model.stream_tts_sync(full_response))
    )

    
    sample_rate = tts_data[0][0]
    audio_chunks = [chunk for _, chunk in tts_data]
    
    # Combine all audio chunks
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        
        # Extract visemes using ForceAlign from the complete audio
        try:
            if viseme_extractor:
                visemes = await viseme_extractor.extract_visemes(full_audio, sample_rate, full_response)
                logger.info(f"Extracted {len(visemes)} visemes using ForceAlign")
            else:
                logger.warning("No viseme extractor available")
                visemes = []
        except Exception as e:
            logger.error(f"Failed to extract visemes: {e}")
            visemes = []
        
        # Format visemes for frontend
        enhanced_visemes = []
        for viseme in visemes:
            enhanced_viseme = {
                "viseme": str(viseme.viseme),
                "start_time": float(viseme.start_time),
                "end_time": float(viseme.end_time),
                "confidence": float(viseme.confidence)
            }
            enhanced_visemes.append(enhanced_viseme)
            logger.info(
                f"🎵 TTS_VISEME: viseme={enhanced_viseme['viseme']}, time={enhanced_viseme['start_time']:.3f}-{enhanced_viseme['end_time']:.3f}")
        
        # Prepare audio chunks for frontend
        audio_chunks_for_frontend = []
        for chunk in audio_chunks:
            audio_chunks_for_frontend.append((sample_rate, chunk.tolist()))
        
        # Buffer for output data
        response_data = {
            "type": "tts_response",
            "text": full_response,
            "audio_chunks": audio_chunks_for_frontend,
            "visemes": enhanced_visemes
        }
        
        # Send all data to frontend at once
        yield AdditionalOutputs(response_data)
        logger.info("Delivered TTS response and visemes in one batch.")
        
        all_visemes = enhanced_visemes
    else:
        logger.warning("No audio chunks generated")
        all_visemes = []
    
    # Process avatar visemes in background
    try:
        def update_avatar_enhanced():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Update avatar with enhanced visemes
                loop.run_until_complete(
                    enhanced_viseme_controller.update_from_ai_visemes(all_visemes)
                )

                loop.close()
            except Exception as e:
                logger.error(f"Failed to update enhanced avatar: {e}")

        # Update avatar in background thread
        threading.Thread(target=update_avatar_enhanced, daemon=True).start()
    except Exception as e:
        logger.error(f"Failed to start avatar update thread: {e}")

    logger.info("Finished enhanced TTS streaming with ForceAlign visemes.")

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