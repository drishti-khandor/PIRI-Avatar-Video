/**
 * Avatar-related types
 */

// Rename BlendShapeValues to BlendShapeValue for single value
export interface BlendShapeValue {
  name: string;
  value: number;
}

// Add BlendShapeValues as a Record type for multiple values
export type BlendShapeValues = Record<string, number>;

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
  NEUTRAL = "neutral",
  HAPPY = "happy",
  SAD = "sad",
  ANGRY = "angry",
  SURPRISED = "surprised",
  DISGUSTED = "disgusted",
  FEARFUL = "fearful",
}

// Add AvatarStateType for the state machine states
export type AvatarStateType = "idle" | "listening" | "speaking";

// Keep the original AvatarState interface for the full state object
export interface AvatarState {
  current_viseme: string;
  current_emotion: EmotionType;
  blend_shapes: Record<string, number>;
  is_speaking: boolean;
  connected_clients: number;
}

export interface AvatarUpdate {
  type: "avatar_update";
  blend_shapes: Record<string, number>;
  viseme?: string;
  emotion?: EmotionType;
  timestamp: number;
}
