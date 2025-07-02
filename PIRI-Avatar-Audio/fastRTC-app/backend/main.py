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

# --- NEW: Import the viseme extractor ---
from viseme_extractor import VisemeExtractor, AdvancedVisemeExtractor

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

try:
    viseme_extractor = AdvancedVisemeExtractor()
    logger.info("Using advanced viseme extractor")
except:
    viseme_extractor = VisemeExtractor()
    logger.info("Using basic viseme extractor")

def echo(audio):
    stt_time = time.time()
    logging.info("Performing STT")
    # --- REPLACED: Use your STT model ---
    text = stt_model.stt(audio)
    if not text:
        logging.info("STT returned empty string")
        return
    logging.info(f"STT response: {text}")

    yield AdditionalOutputs({"type": "stt", "text": text})

    messages.append({"role": "user", "content": text})
    logging.info(f"STT took {time.time() - stt_time} seconds")



    llm_time = time.time()
    # --- REPLACED: Use your Azure OpenAI LLM ---
    try:
        response = openai_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        full_response = response.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        full_response = "[LLM error]"
    logging.info(f"LLM response: {full_response}")
    logging.info(f"LLM took {time.time() - llm_time} seconds")

    yield AdditionalOutputs({"type": "llm", "text": full_response})

    # # --- REPLACED: Use your Kokoro TTS ---
    # logging.info("Starting TTS streaming.")
    # try:
    #     for audio_chunk in tts_model.stream_tts_sync(full_response):
    #         logging.info(f"Audio chunk: {audio_chunk}")
    #         yield audio_chunk
    #
    #     logging.info("Finished TTS streaming.")
    # except Exception as e:
    #     logging.error(f"TTS failed: {e}")
    #
    # messages.append({"role": "assistant", "content": full_response + " "})

    # --- MODIFIED: TTS with viseme extraction ---
    logging.info("Starting TTS streaming with viseme extraction.")
    chunk_index = 0
    accumulated_time = 0.0

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
                for viseme in visemes:
                    viseme.start_time += chunk_start_time
                    viseme.end_time += chunk_start_time

                # Send viseme data to frontend
                viseme_data = {
                    "type": "visemes",
                    "chunk_index": chunk_index,
                    "visemes": [
                        {
                            "viseme": str(v.viseme),
                            "start_time": float(v.start_time),
                            "end_time": float(v.end_time),
                            "confidence": float(v.confidence)
                        }
                        for v in visemes
                    ],
                    "chunk_duration": float(chunk_duration),
                    "chunk_start_time": float(chunk_start_time)
                }

                logging.info(f"Sent visemes for chunk {chunk_index}: {[(v.viseme, v.start_time, v.end_time, v.confidence) for v in visemes]}")
                yield AdditionalOutputs(viseme_data)
                logging.info(f"Sent visemes for chunk {chunk_index}: {[v.viseme for v in visemes]}")

            except Exception as e:
                logging.error(f"Viseme extraction failed for chunk {chunk_index}: {e}")
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

        logging.info("Finished TTS streaming with visemes.")

    except Exception as e:
        logging.error(f"TTS failed: {e}")

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

# --- NEW: Endpoint to get viseme mapping info ---
@app.get("/viseme-info")
async def get_viseme_info():
    """Return viseme mapping information for frontend"""
    return {
        "viseme_mapping": viseme_extractor.phoneme_to_viseme,
        "viseme_descriptions": {
            "0": "Silence",
            "1": "Open vowels (AE, AH, EH, UH)",
            "2": "Open back vowels (AA, AO, AW, OW)",
            "3": "Diphthongs (AY, EY, OY)",
            "4": "R-colored vowels (ER, AX, IX)",
            "5": "Close front vowels (IH, IY)",
            "6": "Close back vowels (UW, UH)",
            "7": "Bilabials (B, P, M)",
            "8": "Labiodentals (F, V)",
            "9": "Dental fricatives (TH, DH)",
            "10": "Alveolars (T, D, N, L)",
            "11": "Sibilants (S, Z)",
            "12": "Post-alveolars (SH, ZH, CH, JH)",
            "13": "Velars (K, G, NG)",
            "14": "Approximants (R, W, Y, HH)"
        }
    }
