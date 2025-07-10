/**
 * Server-Sent Events Service
 * Handles SSE connections for receiving WebRTC updates
 */

import { getApiUrl } from '@/src/config/api.config';

export interface SSEMessage {
  type: 'stt' | 'tts_response' | 'error';
  text?: string;
  audio_b64?: string;
  visemes?: Array<{
    viseme: string;
    phoneme?: string;
    start_time: number;
    end_time: number;
    confidence: number;
    blend_shapes?: Record<string, number>;
    emotion?: string;
  }>;
  emotion?: string;
  audio_duration?: number;
  message?: string;
  timestamp: number;
}

export class SSEService {
  private eventSource: EventSource | null = null;
  private listeners: Map<string, Set<(data: SSEMessage) => void>> = new Map();
  
  constructor() {}

  connect(webrtcId: string): void {
    if (this.eventSource) {
      this.disconnect();
    }

    const url = getApiUrl(`/updates?webrtc_id=${webrtcId}`);
    this.eventSource = new EventSource(url);

    this.eventSource.onopen = () => {
      console.log('SSE connection opened');
      this.emit('connected', { type: 'error', message: 'Connected', timestamp: Date.now() });
    };

    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as SSEMessage;
        console.log('SSE message received:', data.type);
        
        // Emit based on message type
        this.emit(data.type, data);
        this.emit('message', data);
      } catch (error) {
        console.error('Error parsing SSE message:', error);
      }
    };

    this.eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      this.emit('error', { 
        type: 'error', 
        message: 'Connection error', 
        timestamp: Date.now() 
      });
      
      // EventSource will automatically reconnect
    };
  }

  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      console.log('SSE connection closed');
    }
  }

  on(event: string, callback: (data: SSEMessage) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: (data: SSEMessage) => void): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  private emit(event: string, data: SSEMessage): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in SSE listener for ${event}:`, error);
        }
      });
    }
  }

  isConnected(): boolean {
    return this.eventSource !== null && this.eventSource.readyState === EventSource.OPEN;
  }
}
