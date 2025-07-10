"""
WebRTC routes for real-time communication
"""
import logging
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions
from app.core.audio_chat_processor import AudioChatProcessor
from app.core.avatar.viseme_controller import VisemeAnimationController
from app.config.settings import settings


logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances
viseme_controller = VisemeAnimationController()
audio_chat_processor = AudioChatProcessor(viseme_controller)

# Initialize FastRTC stream
stream = Stream(
    ReplyOnPause(
        audio_chat_processor.process_audio_and_respond,
        algo_options=AlgoOptions(
            audio_chunk_duration=settings.audio_chunk_duration,
            started_talking_threshold=settings.started_talking_threshold,
            speech_threshold=settings.speech_threshold
        ),
        model_options=SileroVadOptions(
            threshold=settings.vad_threshold,
            min_speech_duration_ms=settings.min_speech_duration_ms,
            min_silence_duration_ms=settings.min_silence_duration_ms,
            speech_pad_ms=settings.speech_pad_ms,
            max_speech_duration_s=settings.max_speech_duration_s
        )
    ),
    modality="audio",
    mode="send-receive",
    concurrency_limit=5
)


@router.get("/updates")
async def stream_updates(webrtc_id: str):
    """Stream updates for WebRTC connection"""
    logger.info(f"New WebRTC update stream connection: {webrtc_id}")
    
    async def output_stream():
        try:
            logger.debug(f"Starting output stream for WebRTC ID: {webrtc_id}")
            async for output in stream.output_stream(webrtc_id):
                data = output.args[0] if output.args else {}
                logger.debug(f"Sending data to client {webrtc_id}: {data.get('type', 'unknown')}")
                yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            logger.error(f"Error in update stream for {webrtc_id}: {e}", exc_info=True)
            error_data = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        output_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
