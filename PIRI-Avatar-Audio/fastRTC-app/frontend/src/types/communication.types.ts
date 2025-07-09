/**
 * Communication-related types for WebSocket and WebRTC
 */

export interface ChatMessage {
  type: 'stt' | 'llm';
  text: string;
  timestamp?: number;
}

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: number;
  blend_shapes?: Record<string, number>;
  emotion?: string;
}

export interface VRMWebSocketData extends WebSocketMessage {
  blend_shapes?: Record<string, number>;
  emotion?: string;
}

export interface EnhancedVisemeData {
  viseme: string;
  start_time: number;
  end_time: number;
  confidence: number;
  phoneme?: string;
  emotion?: string;
}

export type WebSocketState = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebRTCOptions {
  onConnected?: () => void;
  onDisconnected?: () => void;
  onMessage?: (message: any) => void;
  onAudioStream?: (stream: MediaStream) => void;
  onAudioLevel?: (level: number) => void;
  audioInputDeviceId?: string;
  audioOutputDeviceId?: string;
  webrtcId?: string;
}
