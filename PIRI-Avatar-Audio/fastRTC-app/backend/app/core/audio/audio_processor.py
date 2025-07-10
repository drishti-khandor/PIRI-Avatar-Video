"""
Audio processing with ForceAlign integration for viseme extraction
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
import torch
import torchaudio

# Try to import ForceAlign
try:
    from forcealign import ForceAlign

    FORCEALIGN_AVAILABLE = True
except ImportError:
    FORCEALIGN_AVAILABLE = False
    logging.warning("ForceAlign not available. Using fallback viseme generation.")

from app.models.avatar import VisemeData
from app.config.settings import settings


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing and viseme extraction"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.align_model = None
        self.sample_rate = 16000
        self._initialize_models()

    def _initialize_models(self):
        """Check ForceAlign availability"""
        self.forcealign_available = FORCEALIGN_AVAILABLE
        if self.forcealign_available:
            logger.info("ForceAlign is available for phoneme extraction")
        else:
            logger.warning(
                "ForceAlign not available. Will use fallback viseme generation."
            )

    def _phoneme_to_viseme(self, phoneme: str) -> str:
        """Convert phoneme to viseme"""
        # Comprehensive phoneme to viseme mapping
        phoneme_to_viseme_map = {
            # Vowels
            "AA": "AA",
            "AE": "AE",
            "AH": "AH",
            "AO": "AO",
            "AW": "AW",
            "AY": "AY",
            "EH": "EH",
            "ER": "ER",
            "EY": "EY",
            "IH": "IH",
            "IY": "IY",
            "OW": "OW",
            "OY": "OY",
            "UH": "UH",
            "UW": "UW",
            # Consonants
            "B": "PP",
            "P": "PP",
            "M": "MM",
            "F": "FF",
            "V": "VV",
            "TH": "TH",
            "DH": "TH",
            "D": "DD",
            "T": "TT",
            "N": "NN",
            "L": "LL",
            "G": "KK",
            "K": "KK",
            "NG": "NG",
            "S": "SS",
            "Z": "ZZ",
            "SH": "SH",
            "ZH": "SH",
            "CH": "CH",
            "JH": "JH",
            "Y": "YY",
            "W": "WW",
            "R": "RR",
            "HH": "HH",
            # Silence
            "SIL": "sil",
            "SP": "sil",
            "SPN": "sil",
        }

        # Remove stress numbers and convert to uppercase
        clean_phoneme = "".join(c for c in phoneme if not c.isdigit()).upper()

        return phoneme_to_viseme_map.get(clean_phoneme, "sil")

    def process_audio_chunk(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[torch.Tensor, int]:
        """Process audio chunk and prepare for analysis"""
        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data

        # Ensure correct shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            audio_tensor = resampler(audio_tensor)

        return audio_tensor, self.sample_rate

    async def extract_visemes(
        self, audio_data: np.ndarray, text: str, sample_rate: int = 16000
    ) -> List[VisemeData]:
        """Extract visemes from audio using ForceAlign"""
        if not self.forcealign_available or not text.strip():
            raise RuntimeError("ForceAlign not available or text is empty.")

        try:
            import tempfile
            import soundfile as sf
            import os

            # Create temporary audio file (ForceAlign requires file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Process audio to ensure correct format
                audio_tensor, sr = self.process_audio_chunk(audio_data, sample_rate)
                audio_numpy = audio_tensor.squeeze().numpy()

                # Write audio data to temporary file
                sf.write(temp_audio.name, audio_numpy, sr)
                temp_audio_path = temp_audio.name

            try:
                # Create ForceAlign instance with audio file and transcript
                aligner = ForceAlign(temp_audio_path, text)

                # Perform inference to get alignments
                words = aligner.inference()
                phoneme_alignments = aligner.phoneme_alignments

                visemes = []
                for phoneme_alignment in phoneme_alignments:
                    # Extract phoneme and clean it
                    raw_phoneme = phoneme_alignment.phoneme
                    clean_phoneme = "".join(
                        c for c in raw_phoneme if c.isalpha()
                    ).upper()

                    viseme = self._phoneme_to_viseme(clean_phoneme)

                    viseme_data = VisemeData(
                        viseme=viseme,
                        phoneme=clean_phoneme,
                        start_time=phoneme_alignment.time_start,
                        end_time=phoneme_alignment.time_end,
                        confidence=1.0,
                    )
                    visemes.append(viseme_data)

                # Add silence at the end if needed
                audio_duration = len(audio_data) / sample_rate
                if visemes and visemes[-1].end_time < audio_duration:
                    visemes.append(
                        VisemeData(
                            viseme="sil",
                            phoneme="SIL",
                            start_time=visemes[-1].end_time,
                            end_time=audio_duration,
                            confidence=1.0,
                        )
                    )

                logger.info(f"Extracted {len(visemes)} visemes using ForceAlign")
                return visemes

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Failed to extract visemes with ForceAlign: {e}")
            raise RuntimeError("Failed to extract visemes.")

    def generate_fallback_visemes(self, text: str, duration: float) -> List[VisemeData]:
        """Generate fallback visemes when ForceAlign is not available"""
        # Simple fallback: generate visemes based on text length
        words = text.split()
        if not words:
            return []

        visemes = []
        time_per_word = duration / len(words)
        current_time = 0.0

        for word in words:
            # Simple heuristic: alternate between common visemes
            viseme_sequence = ["AA", "IH", "SS"] * (len(word) // 3 + 1)
            viseme_duration = time_per_word / len(viseme_sequence)

            for viseme in viseme_sequence[: len(word)]:
                visemes.append(
                    VisemeData(
                        viseme=viseme,
                        start_time=current_time,
                        end_time=current_time + viseme_duration,
                        confidence=0.5,  # Lower confidence for fallback
                    )
                )
                current_time += viseme_duration

            # Add brief silence between words
            visemes.append(
                VisemeData(
                    viseme="sil",
                    start_time=current_time,
                    end_time=current_time + 0.05,
                    confidence=1.0,
                )
            )
            current_time += 0.05

        return visemes
    
    def extract_visemes_sync(
        self, audio_data: np.ndarray, text: str, sample_rate: int = 16000
    ) -> List[VisemeData]:
        """Synchronous version of extract_visemes using ForceAlign only"""
        if not self.forcealign_available:
            logger.warning("ForceAlign not available, using fallback")
            return self.generate_fallback_visemes(text, len(audio_data) / sample_rate)
        
        if not text.strip():
            logger.warning("Empty text provided, returning silence viseme")
            return [VisemeData(
                viseme="sil",
                phoneme="SIL",
                start_time=0.0,
                end_time=len(audio_data) / sample_rate,
                confidence=1.0
            )]

        try:
            import tempfile
            import soundfile as sf
            import os
            
            # Check if audio chunk is too small
            audio_duration = len(audio_data) / sample_rate
            if audio_duration < 0.1:  # Skip very small chunks
                logger.warning(f"Audio chunk too small ({audio_duration:.3f}s), using silence viseme")
                return [VisemeData(
                    viseme="sil",
                    phoneme="SIL",
                    start_time=0.0,
                    end_time=audio_duration,
                    confidence=1.0
                )]

            # Create temporary audio file (ForceAlign requires file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Process audio to ensure correct format
                audio_tensor, sr = self.process_audio_chunk(audio_data, sample_rate)
                audio_numpy = audio_tensor.squeeze().numpy()
                
                logger.debug(f"Writing audio chunk: shape={audio_numpy.shape}, sr={sr}, duration={len(audio_numpy)/sr:.3f}s")

                # Write audio data to temporary file
                sf.write(temp_audio.name, audio_numpy, sr)
                temp_audio_path = temp_audio.name

            try:
                # Create ForceAlign instance with audio file and transcript
                aligner = ForceAlign(temp_audio_path, text)

                # Perform inference to get alignments
                words = aligner.inference()
                phoneme_alignments = aligner.phoneme_alignments

                visemes = []
                for phoneme_alignment in phoneme_alignments:
                    # Extract phoneme and clean it
                    raw_phoneme = phoneme_alignment.phoneme
                    clean_phoneme = "".join(
                        c for c in raw_phoneme if c.isalpha()
                    ).upper()

                    viseme = self._phoneme_to_viseme(clean_phoneme)

                    viseme_data = VisemeData(
                        viseme=viseme,
                        phoneme=clean_phoneme,
                        start_time=phoneme_alignment.time_start,
                        end_time=phoneme_alignment.time_end,
                        confidence=1.0,
                    )
                    visemes.append(viseme_data)

                # Add silence at the end if needed
                audio_duration = len(audio_data) / sample_rate
                if visemes and visemes[-1].end_time < audio_duration:
                    visemes.append(
                        VisemeData(
                            viseme="sil",
                            phoneme="SIL",
                            start_time=visemes[-1].end_time,
                            end_time=audio_duration,
                            confidence=1.0,
                        )
                    )

                logger.info(f"Extracted {len(visemes)} visemes using ForceAlign (sync)")
                return visemes

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Failed to extract visemes with ForceAlign (sync): {str(e)}")
            logger.error(f"Audio shape: {audio_data.shape}, sample_rate: {sample_rate}, text: '{text[:50]}...'")
            # Fall back to simple viseme generation
            logger.info("Falling back to simple viseme generation")
            return self.generate_fallback_visemes(text, len(audio_data) / sample_rate)
