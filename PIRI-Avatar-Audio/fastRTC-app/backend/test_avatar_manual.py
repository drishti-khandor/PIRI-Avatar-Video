#!/usr/bin/env python3
"""
Manual Avatar Test Script
Test the avatar animation without audio input
"""

import asyncio
import websockets
import json
import time

async def test_avatar():
    """Test avatar with manual blend shape data"""
    
    # Connect to the avatar WebSocket
    uri = "ws://localhost:8000/ws/avatar"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to avatar WebSocket")
            
            # Test sequence of mouth shapes
            test_shapes = [
                {"Fcl_MTH_A": 1.0, "Fcl_MTH_Large": 0.3},  # Open mouth
                {"Fcl_MTH_E": 1.0, "Fcl_MTH_Small": 0.3},  # E sound
                {"Fcl_MTH_I": 1.0, "Fcl_MTH_Small": 0.6},  # I sound
                {"Fcl_MTH_O": 1.0, "Fcl_MTH_U": 0.4},     # O sound
                {"Fcl_MTH_U": 1.0, "Fcl_MTH_O": 0.4},     # U sound
                {"Fcl_MTH_Close": 1.0},                     # Closed mouth
                {"Fcl_MTH_Neutral": 1.0}                   # Neutral
            ]
            
            for i, shapes in enumerate(test_shapes):
                print(f"🎭 Testing shape {i+1}: {shapes}")
                
                # Send test message (this won't actually work since the server doesn't handle client messages)
                # But we can see if the connection is working
                test_message = {
                    "type": "test",
                    "shapes": shapes
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Wait a bit
                await asyncio.sleep(1.0)
                
            print("✅ Test sequence completed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Avatar WebSocket Connection...")
    asyncio.run(test_avatar())