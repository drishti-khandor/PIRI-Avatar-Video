# PIRI Avatar Video Backend (Refactored)

A high-performance backend server for real-time 3D avatar animation with synchronized audio-visual playback.

## Features

- 🎭 **VRM Avatar Support**: Full support for VRM format with blend shape animations
- 🎯 **Synchronized Playback**: Perfect sync between audio and lip movements
- 🎨 **Smooth Animations**: Cubic easing for natural transitions
- 🗣️ **ForceAlign Integration**: Accurate phoneme extraction for lip-sync
- 🧠 **AI Chat Integration**: Azure OpenAI for intelligent conversations
- 😊 **Emotion Detection**: Automatic emotion analysis and expression
- 🔌 **WebSocket & WebRTC**: Real-time communication protocols
- ⚡ **High Performance**: Optimized for 60fps animations

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│                 │ ←────────────────→ │                 │
│    Frontend     │                    │    Backend      │
│   (Three.js)    │     WebRTC Audio   │   (FastAPI)     │
│                 │ ←────────────────→ │                 │
└─────────────────┘                    └─────────────────┘
         ↓                                      ↓
    VRM Rendering                        Audio Processing
    Blend Shapes                         STT → LLM → TTS
                                         Viseme Extraction
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
```

### 3. Run the Server
```bash
python run.py
```

The server will start on `http://localhost:8001`

## API Documentation

### WebSocket Endpoint
- **URL**: `ws://localhost:8001/api/avatar/ws/avatar`
- **Purpose**: Real-time avatar updates and control

#### Message Format
```json
{
  "type": "avatar_update",
  "blend_shapes": {
    "aa": 0.8,
    "ee": 0.2
  },
  "viseme": "AA",
  "emotion": "happy",
  "timestamp": 1234567890.123
}
```

### REST Endpoints

#### Health Check
```
GET /health
```

#### Trigger Viseme
```
POST /api/avatar/trigger_viseme
Content-Type: application/json

{
  "phoneme": "AA",
  "duration": 0.5,
  "emotion": "neutral"
}
```

#### Avatar Status
```
GET /api/avatar/avatar_status
```

#### Reset Avatar
```
POST /api/avatar/reset_avatar
```

### WebRTC Endpoints

#### Stream Updates (SSE)
```
GET /updates?webrtc_id={id}
```

## Key Components

### Viseme Controller
Manages avatar animations with:
- Smooth blend shape interpolation
- Animation queuing
- WebSocket connection management
- State synchronization

### Audio Processor
Handles:
- ForceAlign phoneme extraction
- Fallback viseme generation
- Audio format conversion
- Timing synchronization

### Unified Processor
Integrates:
- Speech-to-Text (STT)
- Language Model (LLM)
- Text-to-Speech (TTS)
- Viseme extraction
- Emotion detection

## Configuration

All settings in `app/config/settings.py`:

```python
# Server
port = 8001  # Changed from 8000

# Animation
blend_shape_smoothing = 0.3
viseme_transition_time = 0.1

# Audio
audio_chunk_duration = 0.5
vad_threshold = 0.75
```

## Development

### Project Structure
```
app/
├── api/          # API routes and endpoints
├── core/         # Core business logic
├── models/       # Data models
└── config/       # Configuration
```

### Adding New Features

1. **New Viseme Mapping**:
   Edit `app/core/avatar/viseme_controller.py`

2. **New Emotion**:
   Add to `app/models/avatar.py` EmotionType enum

3. **New API Endpoint**:
   Add to `app/api/routes/`

## Troubleshooting

### Common Issues

1. **Port conflict**: Change port in settings.py
2. **Missing models**: Check ForceAlign installation
3. **WebSocket fails**: Verify CORS settings
4. **No audio**: Check microphone permissions

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use GPU**: ForceAlign runs faster on GPU
2. **Optimize models**: Use quantized models when possible
3. **Batch processing**: Process multiple visemes together
4. **Connection pooling**: Reuse WebSocket connections

## License

This project is proprietary. All rights reserved.
