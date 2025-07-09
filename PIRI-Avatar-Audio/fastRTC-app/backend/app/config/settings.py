"""
Application configuration and settings management
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server configuration
    app_name: str = "PIRI Avatar Video Server"
    app_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
    reload: bool = True
    
    # CORS settings
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # WebRTC configuration
    webrtc_ip: Optional[str] = None
    
    # Azure OpenAI configuration
    azure_openai_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_deployment_name: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # Model configuration
    stt_model: str = "default"
    tts_model: str = "kokoro"
    
    # Audio processing settings
    audio_chunk_duration: float = 0.5
    started_talking_threshold: float = 0.1
    speech_threshold: float = 0.03
    
    # VAD (Voice Activity Detection) settings
    vad_threshold: float = 0.75
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1500
    speech_pad_ms: int = 400
    max_speech_duration_s: float = 15.0
    
    # Animation settings
    blend_shape_smoothing: float = 0.3
    viseme_transition_time: float = 0.1
    
    # File paths
    static_dir: str = "static"
    vrm_models_dir: str = "static"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Platform-specific setup
def setup_platform_specific():
    """Configure platform-specific settings"""
    import platform
    import socket
    
    if platform.system() == 'Windows':
        # Get local IP for Windows WebRTC
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            os.environ['WEBRTC_IP'] = local_ip
            settings.webrtc_ip = local_ip
        except Exception:
            settings.webrtc_ip = '127.0.0.1'
        finally:
            s.close()
