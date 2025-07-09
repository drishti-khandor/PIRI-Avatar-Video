"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { WebRTCService } from "@/src/services/webrtc.service";
import { WebSocketService } from "@/src/services/websocket.service";
import { AudioSyncService } from "@/src/services/audio-sync.service";
import { visemeToBlendShapes } from "@/src/utils/viseme-mapping";
import {
  AvatarState,
  BlendShapeValues,
  EmotionState,
} from "@/src/types/avatar.types";
import {
  ChatMessage,
  WebSocketMessage,
} from "@/src/types/communication.types";
  className?: string;
}

export default function AvatarChat({ className = "" }: AvatarChatProps) {
  // Connection states
  const [isWebRTCConnected, setIsWebRTCConnected] = useState(false);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

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
  const audioRef = useRef<HTMLAudioElement>(null);
  const audioSyncRef = useRef<AudioSyncService | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const visemeQueueRef = useRef<any[]>([]);

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocketService("/api/avatar/ws/avatar");

    ws.on("stateChange", (state: string) => {
      setIsWebSocketConnected(state === "connected");
    });

    ws.on("viseme_update", (data: WebSocketMessage) => {
      if (data.blend_shapes) {
        setBlendShapes(data.blend_shapes);
        applyBlendShapesToAvatar(data.blend_shapes);
      }
      if (data.emotion) {
        setCurrentEmotion(data.emotion as EmotionState);
      }
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
        },
        onDisconnected: () => {
          setIsWebRTCConnected(false);
          setIsRecording(false);
        },
        onMessage: (message: ChatMessage) => {
          setMessages((prev) => [...prev, message]);
        },
        onAudioStream: (stream: MediaStream) => {
          if (audioRef.current) {
            audioRef.current.srcObject = stream;
            audioRef.current.play().catch(console.error);
          }
        },
        onAudioLevel: (level: number) => {
          setAudioLevel(level);
        },
      });

      await webrtc.connect();
      webrtcRef.current = webrtc;
    } catch (error) {
      console.error("Failed to start WebRTC:", error);
      setIsWebRTCConnected(false);
    }
  }, []);

  // Stop WebRTC connection
  const stopWebRTC = useCallback(() => {
    if (webrtcRef.current) {
      webrtcRef.current.disconnect();
      webrtcRef.current = null;
    }
    setIsRecording(false);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initialize avatar renderer
  const { applyBlendShapes: applyBlendShapesToAvatar } = useAvatarRenderer({
    canvasRef,
    modelPath: "/models/4thjuly.vrm", // Update this path to your avatar model
  });

  return (
    <div className={`flex h-full w-full ${className}`}>
      {/* Avatar View */}
      <div className="flex-1 relative bg-black">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ touchAction: "none" }}
        />

        {/* Status Indicators */}
        <div className="absolute top-4 left-4 space-y-2">
          <StatusIndicator label="WebSocket" connected={isWebSocketConnected} />
          <StatusIndicator label="WebRTC" connected={isWebRTCConnected} />
        </div>

        {/* Audio Level Indicator */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-gray-800 rounded-full h-2 overflow-hidden">
            <motion.div
              className="bg-green-500 h-full"
              animate={{ width: `${audioLevel * 100}%` }}
              transition={{ duration: 0.1 }}
            />
          </div>
        </div>

        {/* Hidden Audio Element */}
        <audio ref={audioRef} className="hidden" autoPlay />
      </div>

      {/* Chat Panel */}
      <div className="w-96 bg-gray-900 flex flex-col">
        {/* Chat Header */}
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-white font-semibold">Chat</h2>
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

        {/* Controls */}
        <div className="p-4 border-t border-gray-800">
          <button
            onClick={isRecording ? stopWebRTC : startWebRTC}
            className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
              isRecording
                ? "bg-red-600 hover:bg-red-700 text-white"
                : "bg-green-600 hover:bg-green-700 text-white"
            }`}
          >
            {isRecording ? "Stop Recording" : "Start Recording"}
          </button>
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
    <div className="flex items-center space-x-2 bg-gray-800 px-3 py-1 rounded">
      <div
        className={`w-2 h-2 rounded-full ${
          connected ? "bg-green-500" : "bg-red-500"
        }`}
      />
      <span className="text-xs text-gray-300">{label}</span>
    </div>
  );
}
