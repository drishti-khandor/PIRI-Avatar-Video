#!/usr/bin/env python
"""
Run the refactored backend server
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting PIRI Avatar Video Server")
    print("📍 Server will run on: http://localhost:8001")
    print("🎯 Features:")
    print("   ✅ VRM avatar support with blend shapes")
    print("   ✅ Synchronized audio-visual playback")
    print("   ✅ Smooth facial animation transitions")
    print("   ✅ ForceAlign phoneme extraction")
    print("   ✅ Emotion detection and expression")
    print("   ✅ Real-time WebSocket communication")
    print("\n📋 Make sure to:")
    print("   1. Set up your .env file with Azure OpenAI credentials")
    print("   2. Place VRM files in the static/ directory")
    print("   3. Update frontend to connect to port 8001")
    print("\n")
    
    import uvicorn
    from app.config.settings import settings
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
