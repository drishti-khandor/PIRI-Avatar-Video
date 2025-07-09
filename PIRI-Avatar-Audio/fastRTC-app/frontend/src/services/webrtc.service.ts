/**
 * WebRTC Service
 * Handles WebRTC connections for audio streaming
 */

import { API_CONFIG, getApiUrl } from '@/src/config/api.config';
import { WebRTCOptions } from '@/src/types/communication.types';

export class WebRTCService {
  private peerConnection: RTCPeerConnection | null = null;
  private mediaStream: MediaStream | null = null;
  private dataChannel: RTCDataChannel | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private animationFrameId: number | null = null;
  private options: WebRTCOptions;

  constructor(options: WebRTCOptions = {}) {
    this.options = options;
  }

  async connect(): Promise<void> {
    try {
      // Create peer connection
      this.peerConnection = new RTCPeerConnection();

      // Get user media
      this.mediaStream = await this.getUserMedia();
      
      // Setup audio analysis
      this.setupAudioAnalysis();

      // Add tracks to peer connection
      this.mediaStream.getTracks().forEach((track) => {
        if (this.peerConnection && this.mediaStream) {
          this.peerConnection.addTrack(track, this.mediaStream);
        }
      });

      // Handle incoming tracks
      this.peerConnection.addEventListener('track', this.handleTrack.bind(this));

      // Create data channel
      this.dataChannel = this.peerConnection.createDataChannel('text');
      this.dataChannel.addEventListener('message', this.handleDataChannelMessage.bind(this));

      // Create and send offer
      const offer = await this.peerConnection.createOffer();
      await this.peerConnection.setLocalDescription(offer);

      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.WEBRTC.OFFER), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          webrtc_id: this.options.webrtcId,
        }),
      });

      if (!response.ok) {
        throw new Error(`WebRTC offer failed: ${response.statusText}`);
      }

      const serverResponse = await response.json();
      await this.peerConnection.setRemoteDescription(serverResponse);

      this.options.onConnected?.();
    } catch (error) {
      console.error('WebRTC connection error:', error);
      this.disconnect();
      throw error;
    }
  }

  private async getUserMedia(): Promise<MediaStream> {
    const constraints: MediaStreamConstraints = {
      audio: this.options.audioInputDeviceId
        ? { deviceId: { exact: this.options.audioInputDeviceId } }
        : true,
    };

    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error: any) {
      if (error.name === 'NotAllowedError') {
        throw new Error('Microphone access denied. Please allow microphone access.');
      } else if (error.name === 'NotFoundError') {
        throw new Error('No microphone detected. Please connect a microphone.');
      }
      throw error;
    }
  }

  private handleTrack(event: RTCTrackEvent): void {
    if (this.options.onAudioStream) {
      const stream = event.streams[0];
      this.options.onAudioStream(stream);
    }
  }

  private handleDataChannelMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data);
      this.options.onMessage?.(message);
    } catch (error) {
      console.error('Error parsing data channel message:', error);
    }
  }

  private setupAudioAnalysis(): void {
    if (!this.mediaStream || !this.options.onAudioLevel) return;

    try {
      this.audioContext = new AudioContext();
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;

      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      source.connect(this.analyser);

      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);

      this.startAudioLevelAnalysis();
    } catch (error) {
      console.error('Error setting up audio analysis:', error);
    }
  }

  private startAudioLevelAnalysis(): void {
    if (!this.analyser || !this.dataArray || !this.options.onAudioLevel) return;

    let lastUpdateTime = 0;
    const throttleInterval = 100;

    const analyze = () => {
      this.analyser!.getByteFrequencyData(this.dataArray!);

      const currentTime = Date.now();
      if (currentTime - lastUpdateTime > throttleInterval) {
        let sum = 0;
        for (let i = 0; i < this.dataArray!.length; i++) {
          sum += this.dataArray![i];
        }
        const average = sum / this.dataArray!.length / 255;

        this.options.onAudioLevel!(average);
        lastUpdateTime = currentTime;
      }

      this.animationFrameId = requestAnimationFrame(analyze);
    };

    this.animationFrameId = requestAnimationFrame(analyze);
  }

  private stopAudioAnalysis(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.analyser = null;
    this.dataArray = null;
  }

  disconnect(): void {
    this.stopAudioAnalysis();

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.peerConnection) {
      this.peerConnection.close();
      this.peerConnection = null;
    }

    this.dataChannel = null;
    this.options.onDisconnected?.();
  }

  setAudioInputDevice(deviceId: string): void {
    this.options.audioInputDeviceId = deviceId;
    if (this.peerConnection) {
      this.disconnect();
      this.connect();
    }
  }

  setAudioOutputDevice(deviceId: string): void {
    this.options.audioOutputDeviceId = deviceId;
  }
}
