"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { WebRTCService } from "@/src/services/webrtc.service";
import { WebSocketService } from "@/src/services/websocket.service";
import { AudioSyncService } from "@/src/services/audio-sync.service";
import { useAvatarRenderer } from "@/src/hooks/useAvatarRenderer";
import { visemeToBlendShapes, interpolateBlendShapes } from "@/src/utils/viseme-mapping";
import {
  AvatarState,
  BlendShapeValues,
  EmotionState,
} from "@/src/types/avatar.types";
import {
  ChatMessage,
  WebSocketMessage,
} from "@/src/types/communication.types";

interface AvatarChatEnhancedProps {
  className?: string;
}

export default function AvatarChatEnhanced({ className = "" }: AvatarChatEnhancedProps) {
  // Connection states
  const [isWebRTCConnected, setIsWebRTCConnected] = useState(false);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Avatar states
  const [avatarState, setAvatarState] = useState<AvatarState>("idle");
  const [currentEmotion, setCurrentEmotion] = useState<EmotionState>("neutral");
  const [blendShapes, setBlendShapes] = useState<BlendShapeValues>({});
  const [audioLevel, setAudioLevel] = useState(0);

  // Chat messages
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  // Refs
  const webrtcRef = useRef<WebRTCService | null>(null);
  const websocketRef = useRef<WebSocketService | null>(null);
  const audioSyncRef = useRef<AudioSyncService | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const currentBlendShapesRef = useRef<BlendShapeValues>({});
  const targetBlendShapesRef = useRef<BlendShapeValues>({});
  const animationFrameRef = useRef<number | null>(null);

  // Initialize avatar renderer
  const { applyBlendShapes: applyBlendShapesToAvatar, vrmLoaded } = useAvatarRenderer({
    canvasRef,
    modelPath: "/models/4thjuly.vrm",
  });

  // Smooth blend shape interpolation
  const updateBlendShapeAnimation = useCallback(() => {
    const currentShapes = currentBlendShapesRef.current;
    const targetShapes = targetBlendShapesRef.current;
    
    // Interpolate between current and target
    const interpolatedShapes = interpolateBlendShapes(currentShapes, targetShapes, 0.3);
    
    // Apply to avatar
    applyBlendShapesToAvatar(interpolatedShapes);
    setBlendShapes(interpolatedShapes);
    
    // Update current shapes
    currentBlendShapesRef.current = interpolatedShapes;
    
    // Continue animation
    animationFrameRef.current = requestAnimationFrame(updateBlendShapeAnimation);
  }, [applyBlendShapesToAvatar]);

  // Initialize audio sync service
  useEffect(() => {
    const audioSync = new AudioSyncService({
      onVisemeUpdate: (viseme: string, weight: number) => {
        // Convert viseme to blend shapes
        const visemeShapes = visemeToBlendShapes(viseme, weight);
        
        // Update target blend shapes
        targetBlendShapesRef.current = {
          ...targetBlendShapesRef.current,
          ...visemeShapes,
        };
      },
      onAudioStart: () => {
        setIsSpeaking(true);
        setAvatarState("speaking");
      },
      onAudioEnd: () => {
        setIsSpeaking(false);
        setAvatarState("idle");
        // Reset to neutral expression
        targetBlendShapesRef.current = {};
      },
      bufferTime: 50, // Reduced buffer for lower latency
    });

    audioSyncRef.current = audioSync;
    
    // Start blend shape animation loop
    animationFrameRef.current = requestAnimationFrame(updateBlendShapeAnimation);

    return () => {
      audioSync.destroy();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [updateBlendShapeAnimation]);

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocketService("/api/avatar/ws/avatar");

    ws.on("stateChange", (state: string) => {
      setIsWebSocketConnected(state === "connected");
    });

    ws.on("viseme_update", (data: WebSocketMessage) => {
      if (data.blend_shapes) {
        // Update target blend shapes for smooth interpolation
        targetBlendShapesRef.current = {
          ...targetBlendShapesRef.current,
          ...data.blend_shapes,
        };
      }
      if (data.emotion) {
        setCurrentEmotion(data.emotion as EmotionState);
      }
    });

    ws.on("audio_chunk", async (data: any) => {
      if (data.audio && data.visemes && audioSyncRef.current) {
        // Convert base64 audio to ArrayBuffer
        const audioData = Uint8Array.from(atob(data.audio), c => c.charCodeAt(0)).buffer;
        
        // Queue audio with synchronized visemes
        await audioSyncRef.current.queueAudio(audioData, data.visemes);
      }
    });

    ws.on("message", (data: ChatMessage) => {
      setMessages((prev) => [...prev, data]);
    });

    ws.connect().catch(console.error);
    websocketRef.current = ws;

    return () => {
      ws.disconnect();
    };
  }, []);

  // Initialize WebRTC connection
  const startWebRTC = useCallback(async () => {
    try {
      const webrtc = new WebRTCService({
        onConnected: () => {
          setIsWebRTCConnected(true);
          setIsRecording(true);
          setAvatarState("listening");
        },
        onDisconnected: () => {
          setIsWebRTCConnected(false);
          setIsRecording(false);
          setAvatarState("idle");
        },
        onMessage: (message: ChatMessage) => {
          setMessages((prev) => [...prev, message]);
          
          // Handle STT messages
          if (message.type === "stt" && websocketRef.current) {
            // Send to backend for processing
            websocketRef.current.send({
              type: "user_message",
              text: message.text,
            });
          }
        },
        onAudioStream: (stream: MediaStream) => {
          // We don't use the raw audio element anymore
          // Audio is handled by AudioSyncService for better sync
        },
        onAudioLevel: (level: number) => {
          setAudioLevel(level);
          
          // Subtle mouth movement when speaking
          if (isRecording && level > 0.1) {
            targetBlendShapesRef.current = {
              ...targetBlendShapesRef.current,
              "vrc.v_aa": level * 0.3, // Subtle mouth opening based on audio level
            };
          }
        },
      });

      await webrtc.connect();
      webrtcRef.current = webrtc;
    } catch (error) {
      console.error("Failed to start WebRTC:", error);
      setIsWebRTCConnected(false);
    }
  }, [isRecording]);

  // Stop WebRTC connection
  const stopWebRTC = useCallback(() => {
    if (webrtcRef.current) {
      webrtcRef.current.disconnect();
      webrtcRef.current = null;
    }
    setIsRecording(false);
    setAvatarState("idle");
    targetBlendShapesRef.current = {};
  }, []);

  // Toggle recording
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopWebRTC();
    } else {
      startWebRTC();
    }
  }, [isRecording, startWebRTC, stopWebRTC]);

  // Auto-scroll chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Keyboard shortcut for recording (spacebar)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        toggleRecording();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [toggleRecording]);

  return (
    <div className={`flex h-full w-full ${className}`}>
      {/* Avatar View */}
      <div className="flex-1 relative bg-gradient-to-br from-gray-900 to-black">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ touchAction: "none" }}
        />

        {/* Status Indicators */}
        <div className="absolute top-4 left-4 space-y-2">
          <StatusIndicator label="WebSocket" connected={isWebSocketConnected} />
          <StatusIndicator label="WebRTC" connected={isWebRTCConnected} />
          <StatusIndicator label="Avatar" connected={vrmLoaded} />
        </div>

        {/* Avatar State */}
        <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Avatar State</div>
          <div className="text-sm text-white font-medium capitalize">
            {avatarState}
          </div>
          {isSpeaking && (
            <div className="flex items-center gap-2 mt-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs text-green-400">Speaking</span>
            </div>
          )}
        </div>

        {/* Audio Level Visualization */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-gray-800 rounded-full h-2 overflow-hidden">
            <motion.div
              className="bg-gradient-to-r from-blue-500 to-green-500 h-full"
              animate={{ width: `${audioLevel * 100}%` }}
              transition={{ duration: 0.05 }}
            />
          </div>
          {isRecording && (
            <div className="text-xs text-gray-400 mt-1 text-center">
              Listening... (Press spacebar or click mic to stop)
            </div>
          )}
        </div>

        {/* Recording Toggle Button */}
        <button
          onClick={toggleRecording}
          className={`absolute bottom-8 left-1/2 transform -translate-x-1/2 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 ${
            isRecording
              ? "bg-red-500 hover:bg-red-600 scale-110 animate-pulse shadow-lg shadow-red-500/50"
              : "bg-blue-500 hover:bg-blue-600 hover:scale-105 shadow-lg shadow-blue-500/50"
          }`}
        >
          <svg
            className="w-10 h-10 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            {isRecording ? (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            ) : (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            )}
          </svg>
        </button>
      </div>

      {/* Chat Panel */}
      <div className="w-96 bg-gray-900 flex flex-col">
        {/* Chat Header */}
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-white font-semibold">Conversation</h2>
          <div className="text-sm text-gray-400 mt-1">
            Emotion: {currentEmotion}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={`p-3 rounded-lg ${
                  message.type === "stt"
                    ? "bg-blue-900 text-blue-100 ml-8"
                    : "bg-gray-800 text-gray-100 mr-8"
                }`}
              >
                <div className="text-xs text-gray-400 mb-1">
                  {message.type === "stt" ? "You" : "Assistant"}
                </div>
                <div>{message.text}</div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={chatBottomRef} />
        </div>

        {/* Audio Sync Status */}
        {audioSyncRef.current && (
          <div className="p-2 border-t border-gray-800 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>Audio Latency:</span>
              <span>{(audioSyncRef.current.getLatency() * 1000).toFixed(1)}ms</span>
            </div>
            <div className="flex justify-between">
              <span>Buffered:</span>
              <span>{audioSyncRef.current.getBufferedDuration().toFixed(2)}s</span>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="p-4 border-t border-gray-800">
          <div className="text-center text-sm text-gray-400">
            <div>Click the microphone or press spacebar to start talking</div>
            <div className="text-xs mt-1 text-blue-400">
              Enhanced audio sync for smooth lip movement
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Status Indicator Component
interface StatusIndicatorProps {
  label: string;
  connected: boolean;
}

function StatusIndicator({ label, connected }: StatusIndicatorProps) {
  return (
    <div className="flex items-center space-x-2 bg-gray-800/50 backdrop-blur-sm px-3 py-1 rounded">
      <div
        className={`w-2 h-2 rounded-full ${
          connected ? "bg-green-500" : "bg-red-500"
        }`}
      />
      <span className="text-xs text-gray-300">{label}</span>
    </div>
  );
}
