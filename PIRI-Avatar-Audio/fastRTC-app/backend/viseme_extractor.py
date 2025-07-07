import numpy as np
import librosa
from typing import List, Tuple, Dict
import asyncio
from dataclasses import dataclass
from fastrtc import AdditionalOutputs
import logging
logger = logging.getLogger(__name__)



@dataclass
class VisemeData:
    viseme: str
    start_time: float
    end_time: float
    confidence: float = 1.0


class VisemeExtractor:
    """Extract visemes from audio chunks using phoneme-to-viseme mapping"""

    def __init__(self):
        # Phoneme to viseme mapping (Microsoft Speech Platform mapping)
        self.phoneme_to_viseme = {
            # Silence
            'sil': '0', 'sp': '0',

            # Vowels
            'AE': '1', 'AH': '1', 'EH': '1', 'UH': '1',  # Open vowels
            'AA': '2', 'AO': '2', 'AW': '2', 'OW': '2',  # Open back vowels
            'AY': '3', 'EY': '3', 'OY': '3',  # Diphthongs
            'ER': '4', 'AX': '4', 'IX': '4',  # R-colored vowels
            'IH': '5', 'IY': '5',  # Close front vowels
            'UW': '6', 'UH': '6',  # Close back vowels

            # Consonants
            'B': '7', 'P': '7', 'M': '7',  # Bilabials
            'F': '8', 'V': '8',  # Labiodentals
            'TH': '9', 'DH': '9',  # Dental fricatives
            'T': '10', 'D': '10', 'N': '10', 'L': '10',  # Alveolars
            'S': '11', 'Z': '11',  # Sibilants
            'SH': '12', 'ZH': '12', 'CH': '12', 'JH': '12',  # Post-alveolars
            'K': '13', 'G': '13', 'NG': '13',  # Velars
            'R': '14', 'W': '14', 'Y': '14', 'HH': '14',  # Approximants
        }

        # Initialize phoneme recognizer (using a simple energy-based approach)
        # In production, you'd want to use a proper phoneme recognition model
        self.frame_duration = 0.025  # 25ms frames
        self.hop_length = 512

    def extract_features(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract MFCC features from audio chunk"""
        try:
            # Ensure audio is float32 and normalized
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            if len(audio_chunk) == 0:
                return np.array([])

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_chunk,
                sr=sample_rate,
                n_mfcc=13,
                hop_length=self.hop_length
            )
            return mfccs
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.array([])

    def simple_phoneme_detection(self, audio_chunk: np.ndarray, sample_rate: int) -> List[str]:
        """Enhanced phoneme detection based on multiple features"""
        try:
            if len(audio_chunk) == 0:
                return ['sil']

            # Calculate multiple features
            energy = np.sum(audio_chunk ** 2)

            if energy < 1e-5:  # Very quiet
                return ['sil']

            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_chunk, sr=sample_rate, hop_length=self.hop_length
            )
            zcr = librosa.feature.zero_crossing_rate(
                audio_chunk, hop_length=self.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_chunk, sr=sample_rate, hop_length=self.hop_length
            )

            # Get average values
            avg_centroid = np.mean(spectral_centroid) if len(spectral_centroid) > 0 else 0
            avg_zcr = np.mean(zcr) if len(zcr) > 0 else 0
            avg_rolloff = np.mean(spectral_rolloff) if len(spectral_rolloff) > 0 else 0

            # Create frames for temporal analysis
            frame_length = min(1024, len(audio_chunk))
            num_frames = max(1, len(audio_chunk) // frame_length)
            phonemes = []

            for i in range(num_frames):
                start_idx = i * frame_length
                end_idx = min((i + 1) * frame_length, len(audio_chunk))
                frame = audio_chunk[start_idx:end_idx]

                frame_energy = np.sum(frame ** 2)

                if frame_energy < 1e-6:
                    phonemes.append('sil')
                    continue

                # Enhanced classification with more phoneme variety
                if avg_centroid > 4000 and avg_zcr > 0.15:
                    # High frequency fricatives
                    if avg_rolloff > 8000:
                        phonemes.append('S')  # Sibilants
                    else:
                        phonemes.append('F')  # Fricatives
                elif avg_centroid > 3000:
                    # Mid-high frequency consonants
                    if avg_zcr > 0.1:
                        phonemes.append('T')  # Stops
                    else:
                        phonemes.append('SH')  # Affricates
                elif avg_centroid > 1500:
                    # Mid frequency - could be various consonants or vowels
                    if avg_zcr > 0.08:
                        phonemes.append('K')  # Back consonants
                    elif energy > 0.01:
                        phonemes.append('EH')  # Mid vowels
                    else:
                        phonemes.append('N')  # Nasals
                elif avg_zcr < 0.03:
                    # Low ZCR - likely vowels
                    if avg_centroid > 800:
                        phonemes.append('IY')  # High vowels
                    elif avg_centroid > 600:
                        phonemes.append('AE')  # Mid vowels
                    else:
                        phonemes.append('UW')  # Low vowels
                elif avg_zcr < 0.06:
                    # Moderate ZCR - nasals or liquids
                    if avg_centroid > 1000:
                        phonemes.append('R')  # Liquids
                    else:
                        phonemes.append('M')  # Nasals
                else:
                    # Default cases
                    if energy > 0.005:
                        phonemes.append('AH')  # Neutral vowel
                    else:
                        phonemes.append('B')  # Voiced consonant

            # Return at least one phoneme
            return phonemes if phonemes else ['AH']

        except Exception as e:
            print(f"Phoneme detection error: {e}")
            return ['AH']

    def phonemes_to_visemes(self, phonemes: List[str]) -> List[VisemeData]:
        """Convert phonemes to visemes with timing"""
        visemes = []
        duration_per_phoneme = self.frame_duration

        for i, phoneme in enumerate(phonemes):
            viseme_id = self.phoneme_to_viseme.get(phoneme.upper(), '0')
            start_time = i * duration_per_phoneme
            end_time = (i + 1) * duration_per_phoneme

            visemes.append(VisemeData(
                viseme=viseme_id,
                start_time=start_time,
                end_time=end_time,
                confidence=0.8
            ))

        return visemes

    def extract_visemes_from_chunk(
            self,
            audio_chunk: np.ndarray,
            sample_rate: int
    ) -> List[VisemeData]:
        """Extract visemes from a single audio chunk"""
        try:
            # Calculate chunk duration
            chunk_duration = len(audio_chunk) / sample_rate

            # Use simplified approach with fewer, longer visemes
            if len(audio_chunk) == 0 or np.sum(audio_chunk ** 2) < 1e-5:
                return [VisemeData(viseme='0', start_time=0, end_time=chunk_duration)]

            # Create 3-5 visemes per chunk (instead of 100+)
            num_segments = min(5, max(2, int(chunk_duration * 2)))  # 2 visemes per second max
            segment_duration = chunk_duration / num_segments

            visemes = []
            for i in range(num_segments):
                start_idx = int((i / num_segments) * len(audio_chunk))
                end_idx = int(((i + 1) / num_segments) * len(audio_chunk))
                segment = audio_chunk[start_idx:end_idx]

                # Get viseme for this segment
                viseme_id = self.analyze_segment_for_viseme(segment, sample_rate)

                visemes.append(VisemeData(
                    viseme=viseme_id,
                    start_time=i * segment_duration,
                    end_time=(i + 1) * segment_duration,
                    confidence=0.8
                ))

            logger.info(f"🎤 EXTRACTED_VISEMES: {[(v.viseme, v.start_time, v.end_time) for v in visemes]}")

            return visemes

        except Exception as e:
            print(f"Viseme extraction error: {e}")
            chunk_duration = len(audio_chunk) / sample_rate if len(audio_chunk) > 0 else 0.1
            return [VisemeData(viseme='0', start_time=0, end_time=chunk_duration)]

    def analyze_segment_for_viseme(self, segment: np.ndarray, sample_rate: int) -> str:
        """Analyze a segment and return appropriate viseme with better variety"""
        try:
            if len(segment) == 0:
                # ADD LOGGING HERE:
                print(f"🔇 AUDIO_ANALYSIS: Empty segment -> viseme='0' (silence)")
                return '0'

            energy = np.sum(segment ** 2)
            if energy < 1e-6:
                # ADD LOGGING HERE:
                print(f"🔇 AUDIO_ANALYSIS: Low energy ({energy:.2e}) -> viseme='0' (silence)")
                return '0'  # Silence

            # Calculate features
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)
            zcr = librosa.feature.zero_crossing_rate(segment)

            avg_centroid = np.mean(spectral_centroid) if len(spectral_centroid) > 0 else 0
            avg_zcr = np.mean(zcr) if len(zcr) > 0 else 0

            # ADD LOGGING HERE - Log the audio features:
            print(f"🔊 AUDIO_FEATURES: energy={energy:.4f}, centroid={avg_centroid:.0f}Hz, zcr={avg_zcr:.3f}")

            # Use random selection within reasonable ranges to get variety
            import random

            # High frequency sounds
            if avg_centroid > 3500:
                result = random.choice(['11', '12'])  # Sibilants or post-alveolars
                # ADD LOGGING HERE:
                print(f"📈 HIGH_FREQ: centroid={avg_centroid:.0f} -> viseme='{result}' (sibilants/post-alveolars)")
                return result
            elif avg_centroid > 2500:
                result = random.choice(['8', '9', '10'])  # Fricatives or alveolars
                # ADD LOGGING HERE:
                print(f"📊 MID_HIGH_FREQ: centroid={avg_centroid:.0f} -> viseme='{result}' (fricatives/alveolars)")
                return result
            elif avg_centroid > 1500:
                if avg_zcr > 0.1:
                    result = random.choice(['10', '13'])  # Consonants
                    # ADD LOGGING HERE:
                    print(
                        f"🔄 MID_FREQ_HIGH_ZCR: centroid={avg_centroid:.0f}, zcr={avg_zcr:.3f} -> viseme='{result}' (consonants)")
                    return result
                else:
                    result = random.choice(['1', '3', '5'])  # Vowels
                    # ADD LOGGING HERE:
                    print(
                        f"🎵 MID_FREQ_LOW_ZCR: centroid={avg_centroid:.0f}, zcr={avg_zcr:.3f} -> viseme='{result}' (vowels)")
                    return result
            elif avg_centroid > 800:
                if energy > 0.01:
                    result = random.choice(['1', '2', '4'])  # Mid/back vowels
                    # ADD LOGGING HERE:
                    print(
                        f"🔉 LOW_MID_FREQ_HIGH_ENERGY: centroid={avg_centroid:.0f}, energy={energy:.4f} -> viseme='{result}' (mid/back vowels)")
                    return result
                else:
                    result = random.choice(['7', '14'])  # Bilabials or approximants
                    # ADD LOGGING HERE:
                    print(
                        f"🤐 LOW_MID_FREQ_LOW_ENERGY: centroid={avg_centroid:.0f}, energy={energy:.4f} -> viseme='{result}' (bilabials/approximants)")
                    return result
            else:
                if avg_zcr < 0.05:
                    result = random.choice(['2', '6'])  # Back vowels
                    # ADD LOGGING HERE:
                    print(
                        f"📉 LOW_FREQ_LOW_ZCR: centroid={avg_centroid:.0f}, zcr={avg_zcr:.3f} -> viseme='{result}' (back vowels)")
                    return result
                else:
                    result = random.choice(['7', '13'])  # Bilabials or velars
                    # ADD LOGGING HERE:
                    print(
                        f"📉 LOW_FREQ_HIGH_ZCR: centroid={avg_centroid:.0f}, zcr={avg_zcr:.3f} -> viseme='{result}' (bilabials/velars)")
                    return result

        except Exception as e:
            print(f"❌ Segment analysis error: {e}")
            # ADD LOGGING HERE:
            print(f"🔄 FALLBACK: Error occurred -> viseme='1' (default neutral vowel)")
            return '1'  # Default to neutral vowel

    async def extract_visemes_async(
            self,
            audio_chunk: np.ndarray,
            sample_rate: int
    ) -> List[VisemeData]:
        """Async version for non-blocking extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.extract_visemes_from_chunk,
            audio_chunk,
            sample_rate
        )


# Alternative: Using a more sophisticated phoneme recognizer
class AdvancedVisemeExtractor(VisemeExtractor):
    """Uses wav2vec2 or similar model for better phoneme recognition"""

    def __init__(self):
        super().__init__()
        try:
            # Try to load a phoneme recognition model
            # You can use wav2vec2, whisper, or other models
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

            self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
                "facebook/wav2vec2-large-960h-lv60-self"
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-960h-lv60-self"
            )
            self.use_advanced = True

        except ImportError:
            print("Advanced models not available, using simple extraction")
            self.use_advanced = False

    def advanced_phoneme_detection(
            self,
            audio_chunk: np.ndarray,
            sample_rate: int
    ) -> List[str]:
        """Use wav2vec2 for phoneme detection"""
        if not self.use_advanced:
            return self.simple_phoneme_detection(audio_chunk, sample_rate)

        try:
            import torch

            # Resample if needed (wav2vec2 expects 16kHz)
            if sample_rate != 16000:
                audio_chunk = librosa.resample(
                    audio_chunk,
                    orig_sr=sample_rate,
                    target_sr=16000
                )

            # Get model predictions
            inputs = self.tokenizer(
                audio_chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = self.model(inputs.input_values).logits

            # Get predicted tokens
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.tokenizer.decode(predicted_ids[0])

            # Convert transcription to phonemes (simplified)
            words = transcription.split()
            phonemes = []
            for word in words:
                # Simple mapping - in practice you'd use a proper G2P system
                phonemes.extend(self.word_to_phonemes(word))

            return phonemes if phonemes else ['sil']

        except Exception as e:
            print(f"Advanced phoneme detection error: {e}")
            return self.simple_phoneme_detection(audio_chunk, sample_rate)

    def word_to_phonemes(self, word: str) -> List[str]:
        """Simple word to phoneme conversion"""
        # This is a very basic implementation
        # In practice, use a proper G2P (Grapheme-to-Phoneme) system
        phoneme_map = {
            'hello': ['HH', 'AH', 'L', 'OW'],
            'world': ['W', 'ER', 'L', 'D'],
            'the': ['DH', 'AH'],
            'and': ['AH', 'N', 'D'],
            'is': ['IH', 'Z'],
            'a': ['AH'],
        }
        return phoneme_map.get(word.lower(), ['AH'])  # Default vowel