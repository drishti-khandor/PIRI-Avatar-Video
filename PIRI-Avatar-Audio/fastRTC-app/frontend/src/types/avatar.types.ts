/**
 * Avatar-related types
 */

export interface BlendShape {
  name: string;
  value: number;
}

export interface VisemeData {
  viseme: string;
  phoneme?: string;
  start_time: number;
  end_time: number;
  confidence: number;
  blend_shapes?: Record<string, number>;
  emotion?: EmotionType;
}

export enum EmotionType {
  NEUTRAL = 'neutral',
  HAPPY = 'happy',
  SAD = 'sad',
  ANGRY = 'angry',
  SURPRISED = 'surprised',
  DISGUSTED = 'disgusted',
  FEARFUL = 'fearful',
}

export interface AvatarState {
  current_viseme: string;
  current_emotion: EmotionType;
  blend_shapes: Record<string, number>;
  is_speaking: boolean;
  connected_clients: number;
}

export interface AvatarUpdate {
  type: 'avatar_update';
  blend_shapes: Record<string, number>;
  viseme?: string;
  emotion?: EmotionType;
  timestamp: number;
}
