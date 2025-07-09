/**
 * Viseme to VRM blend shape mapping utilities
 */

import { BlendShapeValues } from '@/src/types/avatar.types';

// Standard viseme to VRM blend shape mappings
const VISEME_TO_BLEND_SHAPE_MAP: Record<string, string> = {
  // Viseme mappings based on common phonemes
  'sil': 'neutral',
  'aa': 'aa',
  'ae': 'aa',
  'ah': 'aa',
  'ao': 'o',
  'aw': 'o',
  'ay': 'aa',
  'b': 'bmp',
  'ch': 'ch',
  'd': 'dd',
  'dh': 'th',
  'eh': 'e',
  'er': 'e',
  'ey': 'e',
  'f': 'ff',
  'g': 'kk',
  'hh': 'nn',
  'ih': 'i',
  'iy': 'i',
  'jh': 'ch',
  'k': 'kk',
  'l': 'nn',
  'm': 'bmp',
  'n': 'nn',
  'ng': 'nn',
  'ow': 'o',
  'oy': 'o',
  'p': 'bmp',
  'r': 'rr',
  's': 'ss',
  'sh': 'ch',
  't': 'dd',
  'th': 'th',
  'uh': 'u',
  'uw': 'u',
  'v': 'ff',
  'w': 'u',
  'y': 'i',
  'z': 'ss',
  'zh': 'ch',
};

// VRM standard blend shape names
export const VRM_BLEND_SHAPES = {
  // Vowels
  aa: 'vrc.v_aa',
  i: 'vrc.v_ih',
  u: 'vrc.v_ou',
  e: 'vrc.v_e',
  o: 'vrc.v_oh',
  
  // Consonants
  bmp: 'vrc.v_bmp',
  ch: 'vrc.v_ch',
  dd: 'vrc.v_dd',
  ff: 'vrc.v_ff',
  kk: 'vrc.v_kk',
  nn: 'vrc.v_nn',
  pp: 'vrc.v_pp',
  rr: 'vrc.v_rr',
  ss: 'vrc.v_ss',
  th: 'vrc.v_th',
  
  // Expressions
  neutral: 'neutral',
  happy: 'happy',
  sad: 'sad',
  angry: 'angry',
  surprised: 'surprised',
  
  // Blink
  blink: 'blink',
  blinkLeft: 'blinkLeft',
  blinkRight: 'blinkRight',
};

/**
 * Convert viseme to VRM blend shape values
 */
export function visemeToBlendShapes(
  viseme: string,
  intensity: number = 1.0
): BlendShapeValues {
  const blendShapes: BlendShapeValues = {};
  
  // Map viseme to blend shape name
  const mappedShape = VISEME_TO_BLEND_SHAPE_MAP[viseme.toLowerCase()];
  if (!mappedShape || mappedShape === 'neutral') {
    return blendShapes;
  }
  
  // Get VRM blend shape name
  const vrmShapeName = VRM_BLEND_SHAPES[mappedShape as keyof typeof VRM_BLEND_SHAPES];
  if (vrmShapeName) {
    blendShapes[vrmShapeName] = intensity;
  }
  
  return blendShapes;
}

/**
 * Interpolate between two blend shape states
 */
export function interpolateBlendShapes(
  from: BlendShapeValues,
  to: BlendShapeValues,
  factor: number
): BlendShapeValues {
  const result: BlendShapeValues = {};
  
  // Get all unique keys
  const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
  
  for (const key of allKeys) {
    const fromValue = from[key] || 0;
    const toValue = to[key] || 0;
    result[key] = fromValue + (toValue - fromValue) * factor;
  }
  
  return result;
}

/**
 * Apply emotion to blend shapes
 */
export function applyEmotionToBlendShapes(
  blendShapes: BlendShapeValues,
  emotion: string,
  intensity: number = 0.5
): BlendShapeValues {
  const result = { ...blendShapes };
  
  // Add emotion blend shape
  const emotionShape = VRM_BLEND_SHAPES[emotion as keyof typeof VRM_BLEND_SHAPES];
  if (emotionShape) {
    result[emotionShape] = intensity;
  }
  
  return result;
}
