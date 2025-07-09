# Backend Naming Changes Reference

This document lists all the naming changes made during the refactoring to make the code more descriptive and maintainable.

## Class Names

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `EnhancedVisemeController` | `VisemeAnimationController` | Controls viseme animations and WebSocket connections |
| `UnifiedProcessor` | `AudioChatProcessor` | Processes audio through STT → LLM → TTS → Visemes pipeline |

## Variable Names

| Old Name | New Name | Location |
|----------|----------|----------|
| `enhanced_viseme_controller` | `viseme_controller` | Throughout the codebase |
| `unified_processor` | `audio_chat_processor` | `webrtc.py` |
| `vroid_blend_shapes` | `viseme_to_blend_shape_mapping` | `VisemeAnimationController` |
| `enhanced_vrm_support` | `vrm_support` | Health check endpoint |

## Method Names

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `_initialize_vroid_mappings()` | `_initialize_viseme_mappings()` | Initialize blend shape mappings |
| `_interpolate_blend_shapes()` | `_calculate_interpolated_blend_shapes()` | Calculate interpolation between blend shapes |
| `_transition_to_neutral()` | `_animate_to_neutral_expression()` | Animate avatar to neutral expression |

## File Names

| Old Path | New Path |
|----------|----------|
| `app/core/unified_processor.py` | `app/core/audio_chat_processor.py` |

## Parameter Names

| Old Name | New Name | Location |
|----------|----------|----------|
| `factor` | `interpolation_factor` | `_calculate_interpolated_blend_shapes()` |

## Benefits of These Changes

1. **Clarity**: Names now clearly describe what each component does
2. **Consistency**: Removed "enhanced" and "unified" prefixes that didn't add meaning
3. **Searchability**: More specific names make it easier to find code
4. **Maintainability**: New developers can understand the codebase faster
5. **Self-documenting**: Code is more self-explanatory

## Usage Examples

### Before:
```python
enhanced_viseme_controller = EnhancedVisemeController()
unified_processor = UnifiedProcessor(enhanced_viseme_controller)
```

### After:
```python
viseme_controller = VisemeAnimationController()
audio_chat_processor = AudioChatProcessor(viseme_controller)
```

The new names clearly indicate:
- `viseme_controller` handles viseme animations
- `audio_chat_processor` processes audio and chat interactions
