/**
 * Audio Synchronization Service for smooth playback and lip sync
 */

export interface AudioSyncOptions {
  onVisemeUpdate?: (viseme: string, weight: number) => void;
  onAudioStart?: () => void;
  onAudioEnd?: () => void;
  bufferTime?: number; // ms to buffer before playback
}

export class AudioSyncService {
  private audioContext: AudioContext | null = null;
  private audioQueue: Array<{ buffer: AudioBuffer; visemes: any[] }> = [];
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private startTime = 0;
  private pauseTime = 0;
  private visemeTimers: number[] = [];
  private options: AudioSyncOptions;

  constructor(options: AudioSyncOptions = {}) {
    this.options = {
      bufferTime: 100, // 100ms buffer by default
      ...options,
    };
  }

  async initialize() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    // Resume context if suspended
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  async queueAudio(audioData: ArrayBuffer, visemes: any[]) {
    if (!this.audioContext) {
      await this.initialize();
    }

    try {
      const audioBuffer = await this.audioContext!.decodeAudioData(audioData.slice(0));
      this.audioQueue.push({ buffer: audioBuffer, visemes });

      // Start playback if not already playing
      if (!this.isPlaying) {
        this.startPlayback();
      }
    } catch (error) {
      console.error('Error decoding audio:', error);
    }
  }

  private async startPlayback() {
    if (this.isPlaying || this.audioQueue.length === 0) return;

    this.isPlaying = true;
    this.options.onAudioStart?.();

    while (this.audioQueue.length > 0) {
      const { buffer, visemes } = this.audioQueue.shift()!;
      await this.playAudioWithVisemes(buffer, visemes);
    }

    this.isPlaying = false;
    this.options.onAudioEnd?.();
  }

  private playAudioWithVisemes(buffer: AudioBuffer, visemes: any[]): Promise<void> {
    return new Promise((resolve) => {
      if (!this.audioContext) {
        resolve();
        return;
      }

      // Create audio source
      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);

      // Schedule viseme updates with audio timing
      const audioStartTime = this.audioContext.currentTime;
      this.scheduleVisemes(visemes, audioStartTime);

      // Start audio playback
      source.start(audioStartTime);
      this.currentSource = source;
      this.startTime = audioStartTime;

      source.onended = () => {
        this.clearVisemeTimers();
        this.currentSource = null;
        resolve();
      };
    });
  }

  private scheduleVisemes(visemes: any[], audioStartTime: number) {
    this.clearVisemeTimers();

    visemes.forEach((viseme) => {
      // Calculate exact timing relative to audio context time
      const visemeTime = audioStartTime + viseme.start_time;
      const currentTime = this.audioContext!.currentTime;
      const delay = Math.max(0, (visemeTime - currentTime) * 1000);

      const timer = window.setTimeout(() => {
        this.options.onVisemeUpdate?.(viseme.viseme, viseme.confidence || 1.0);
      }, delay);

      this.visemeTimers.push(timer);
    });
  }

  private clearVisemeTimers() {
    this.visemeTimers.forEach(timer => clearTimeout(timer));
    this.visemeTimers = [];
  }

  pause() {
    if (this.currentSource && this.audioContext) {
      this.pauseTime = this.audioContext.currentTime;
      this.currentSource.stop();
      this.clearVisemeTimers();
      this.isPlaying = false;
    }
  }

  resume() {
    if (!this.isPlaying && this.audioQueue.length > 0) {
      this.startPlayback();
    }
  }

  clear() {
    this.pause();
    this.audioQueue = [];
    this.clearVisemeTimers();
  }

  destroy() {
    this.clear();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  getLatency(): number {
    if (!this.audioContext) return 0;
    // Returns the audio context's output latency in seconds
    return (this.audioContext as any).outputLatency || 0.01;
  }

  getBufferedDuration(): number {
    return this.audioQueue.reduce((total, item) => total + item.buffer.duration, 0);
  }
}
