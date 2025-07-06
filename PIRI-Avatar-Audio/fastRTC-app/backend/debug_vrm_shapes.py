#!/usr/bin/env python3
"""
Debug VRM Blend Shapes
Check what blend shapes are available in the VRM file
"""

import requests
import json

def test_avatar_endpoint():
    """Test the avatar test endpoint"""
    try:
        print("🧪 Testing avatar endpoint...")
        response = requests.post("http://localhost:8000/test_avatar")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Test successful:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Test failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def check_avatar_status():
    """Check avatar status"""
    try:
        print("📊 Checking avatar status...")
        response = requests.get("http://localhost:8000/avatar_status")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Status:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Status check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Status check failed: {e}")

def check_connections():
    """Check WebSocket connections"""
    try:
        print("🔗 Checking connections...")
        response = requests.get("http://localhost:8000/debug/connections")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Connections:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Connection check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection check failed: {e}")

def test_emotion():
    """Test emotion setting"""
    try:
        print("😊 Testing emotion...")
        response = requests.post(
            "http://localhost:8000/set_emotion",
            json={"emotion": "happy"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Emotion set:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Emotion test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Emotion test failed: {e}")

if __name__ == "__main__":
    print("🔍 VRM Debug Tool")
    print("=" * 50)
    
    check_avatar_status()
    print()
    
    check_connections()
    print()
    
    test_emotion()
    print()
    
    test_avatar_endpoint()