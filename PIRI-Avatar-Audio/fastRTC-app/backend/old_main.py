import fastapi
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions , AdditionalOutputs
from fastrtc.utils import audio_to_bytes

import logging
import time
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import platform
import os
import socket

# --- NEW: Azure OpenAI and Kokoro TTS imports ---
from dotenv import load_dotenv
from fastrtc import get_stt_model, get_tts_model
from openai import AzureOpenAI

# --- ENV SETUP for Azure OpenAI ---
load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if platform.system() == 'Windows':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    os.environ['WEBRTC_IP'] = local_ip

# --- SYSTEM PROMPT ---
sys_prompt = """
You are a helpful assistant.
"""

messages = [{"role": "system", "content": sys_prompt}]

# --- NEW: Azure OpenAI and Kokoro TTS/STT clients ---
if not all([azure_endpoint, api_key, deployment_name]):
    logger.error("Missing Azure OpenAI environment variables.")
    raise ValueError("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME in your .env file.")

openai_client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version,
)

stt_model = get_stt_model()
tts_model = get_tts_model(model="kokoro")



def convert_to_pcm(audio_chunk_bytes):
    """Convert audio bytes (WAV or similar) to PCM16 mono NumPy array at 16kHz."""
    from pydub import AudioSegment
    import numpy as np
    import io

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_chunk_bytes), format="wav")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        return samples
    except Exception as e:
        logger.error(f"Failed to convert audio chunk to PCM: {e}")
        return None


def get_visemes_from_audio(pcm_array):
    """Mock viseme extractor — replace with real OVRLipSync call."""
    if pcm_array is None:
        return {}
    # Mock viseme weight output
    return {"aa": 0.5, "ih": 0.2, "oh": 0.1}


def float32_to_pcm_int16(float_audio):
    """
    Converts float32 numpy audio array in range [-1.0, 1.0] to int16 PCM.
    """
    if float_audio is None:
        return None
    try:
        # Clip values to prevent wrapping
        int16_audio = np.clip(float_audio, -1.0, 1.0)
        int16_audio = (int16_audio * 32767).astype(np.int16)
        return int16_audio
    except Exception as e:
        logger.error(f"Failed to convert float32 audio to PCM16: {e}")
        return None


def echo(audio):
    stt_time = time.time()
    logging.info("Performing STT")
    text = stt_model.stt(audio)
    if not text:
        logging.info("STT returned empty string")
        return
    logging.info(f"STT response: {text}")
    yield AdditionalOutputs({"type": "stt", "text": text})

    messages.append({"role": "user", "content": text})
    logging.info(f"STT took {time.time() - stt_time:.2f}s")

    llm_time = time.time()
    try:
        response = openai_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        full_response = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        full_response = "[LLM error]"
    logging.info(f"LLM response: {full_response}")
    logging.info(f"LLM took {time.time() - llm_time:.2f}s")

    yield AdditionalOutputs({"type": "llm", "text": full_response})

    logging.info("Starting TTS streaming + viseme extraction.")
    try:
        for i, audio_chunk in enumerate(tts_model.stream_tts_sync(full_response)):
            logger.info(f"Audio chunk {i} - type: {type(audio_chunk)}, length: {len(audio_chunk)}")

            if isinstance(audio_chunk, tuple) and len(audio_chunk) == 2:
                sample_rate, float_audio = audio_chunk
                logger.debug(
                    f"Sample rate: {sample_rate}, Audio dtype: {type(float_audio)}, shape: {getattr(float_audio, 'shape', None)}")

                # Convert float32 -> int16 PCM

                pcm_data = float32_to_pcm_int16(float_audio)
                viseme_data = get_visemes_from_audio(pcm_data)

                logger.debug(
                    f"PCM shape: {pcm_data.shape}, dtype: {pcm_data.dtype}, max: {np.max(pcm_data)}, min: {np.min(pcm_data)}")

                logger.info(f"Generated visemes: {viseme_data}")


                # Yield visemes to frontend
                yield AdditionalOutputs({"type": "viseme", "visemes": viseme_data})
            else:
                logger.warning(f"Unexpected audio_chunk format: {audio_chunk}")

            # Still yield original audio for playback
            yield audio_chunk

        logging.info("Finished TTS streaming.")
    except Exception as e:
        logging.error(f"TTS or viseme streaming failed: {e}")

    messages.append({"role": "assistant", "content": full_response + " "})

# --- Everything else unchanged below this line ---
stream = Stream(ReplyOnPause(
    echo,
    algo_options=AlgoOptions(
        audio_chunk_duration=0.5,
        started_talking_threshold=0.1,
        speech_threshold=0.03
    ),
    model_options=SileroVadOptions(
        threshold=0.75,
        min_speech_duration_ms=250,
        min_silence_duration_ms=1500,
        speech_pad_ms=400,
        max_speech_duration_s=15
    )),
    modality="audio",
    mode="send-receive"
)

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream.mount(app)

@app.get("/reset")
async def reset():
    global messages
    logging.info("Resetting chat")
    messages = [{"role": "system", "content": sys_prompt}]
    return {"status": "success"}



@app.get("/updates")
async def stream_updates(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            # Output is the AdditionalOutputs instance
            # Serialize as JSON string for frontend
            import json
            yield f"data: {json.dumps(output.args[0])}\n\n"
    return StreamingResponse(output_stream(), media_type="text/event-stream")
