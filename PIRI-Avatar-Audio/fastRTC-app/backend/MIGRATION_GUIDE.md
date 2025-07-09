# Backend Refactoring Migration Guide

## Overview
The backend has been completely refactored to run on port 8001 with improved organization, synchronization, and bug fixes.

## Key Changes

### 1. Port Change
- **Old**: Port 8000
- **New**: Port 8001
- **Frontend Update Required**: Update all API calls to use `http://localhost:8001`

### 2. Directory Structure
```
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── avatar.py     # Avatar endpoints
│   │       └── webrtc.py     # WebRTC endpoints
│   ├── core/
│   │   ├── avatar/
│   │   │   └── viseme_controller.py
│   │   ├── audio/
│   │   │   └── audio_processor.py
│   │   ├── ai/
│   │   │   └── llm_client.py
│   │   └── unified_processor.py
│   ├── models/
│   │   └── avatar.py
│   └── config/
│       └── settings.py
└── run.py
```

### 3. API Endpoint Changes

#### Avatar Endpoints
- `/trigger_viseme` → `/api/avatar/trigger_viseme`
- `/ws/avatar` → `/api/avatar/ws/avatar`
- `/avatar_status` → `/api/avatar/avatar_status`
- `/reset_avatar` → `/api/avatar/reset_avatar`

#### WebRTC Endpoints
- `/updates` → `/updates` (unchanged)
- `/webrtc/offer` → `/webrtc/offer` (unchanged)

### 4. WebSocket Communication
- Single unified WebSocket at `/api/avatar/ws/avatar`
- Handles all avatar updates, blend shapes, and control messages
- Automatic client cleanup on disconnect

### 5. Synchronization Improvements
- Audio and animation are now properly synchronized
- Smooth transitions using cubic easing
- Animation queue prevents dropped frames
- Proper timing based on actual audio duration

### 6. Bug Fixes
- Fixed memory leaks in WebSocket connections
- Fixed race conditions in animation scheduling
- Fixed error handling and recovery
- Fixed blend shape interpolation

### 7. Configuration
All settings are now centralized in `app/config/settings.py`:
- Server configuration
- Audio processing parameters
- Animation settings
- AI configuration

### 8. Environment Variables
Create a `.env` file in the backend directory:
```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Running the New Backend

### Option 1: Using run.py
```bash
cd backend
python run.py
```

### Option 2: Using uvicorn directly
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Frontend Updates Required

### 1. Update WebSocket Connection
```javascript
// Old
const ws = new WebSocket('ws://localhost:8000/ws/avatar');

// New
const ws = new WebSocket('ws://localhost:8001/api/avatar/ws/avatar');
```

### 2. Update API Calls
```javascript
// Old
fetch('http://localhost:8000/trigger_viseme', {...})

// New
fetch('http://localhost:8001/api/avatar/trigger_viseme', {...})
```

### 3. Update SSE Connection
```javascript
// Old
const eventSource = new EventSource('http://localhost:8000/updates?webrtc_id=' + id);

// New
const eventSource = new EventSource('http://localhost:8001/updates?webrtc_id=' + id);
```

### 4. Update WebRTC Offer
```javascript
// Old
fetch('http://localhost:8000/webrtc/offer', {...})

// New
fetch('http://localhost:8001/webrtc/offer', {...})
```

## Deprecation Notice
The following files are deprecated and should not be used:
- `unified_server.py`
- `advanced_vroid_viseme_system.py`
- `vroid_viseme_integration.py`
- `forcealign_demo.py`

## Testing the Refactored Backend

1. Check health endpoint:
```bash
curl http://localhost:8001/health
```

2. Test WebSocket connection:
```javascript
const ws = new WebSocket('ws://localhost:8001/api/avatar/ws/avatar');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

3. Trigger a test viseme:
```bash
curl -X POST http://localhost:8001/api/avatar/trigger_viseme \
  -H "Content-Type: application/json" \
  -d '{"phoneme": "AA", "duration": 0.5}'
```

## Benefits of Refactoring

1. **Better Organization**: Code is now modular and easier to maintain
2. **Improved Performance**: Eliminated redundant processing
3. **Better Synchronization**: Audio and visuals are properly synced
4. **Error Resilience**: Better error handling and recovery
5. **Easier Testing**: Modular structure allows unit testing
6. **Configuration Management**: Centralized settings
7. **Type Safety**: Proper data validation with Pydantic models

## Troubleshooting

### Port Already in Use
If port 8001 is already in use:
1. Change port in `app/config/settings.py`
2. Or set environment variable: `PORT=8002`

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### WebSocket Connection Fails
1. Check CORS settings in `settings.py`
2. Ensure frontend is using correct port (8001)
3. Check browser console for errors

### Audio Not Playing
1. Verify TTS model is loaded
2. Check audio format compatibility
3. Ensure proper audio permissions in browser
