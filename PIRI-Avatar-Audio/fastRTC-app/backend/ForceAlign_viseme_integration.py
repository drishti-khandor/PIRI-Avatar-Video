"""
ForceAlign-based Viseme Extraction
This module provides viseme extraction using the ForceAlign library for phoneme-level alignment.
"""

import asyncio
from ctypes import alignment
import logging
import numpy as np
import tempfile
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf

# ForceAlign imports
try:
    from forcealign import ForceAlign
    FORCE_ALIGN_AVAILABLE = True
except ImportError:
    FORCE_ALIGN_AVAILABLE = False
    logging.warning("ForceAlign not available. Please install it with: pip install ForceAlign")

logger = logging.getLogger(__name__)

@dataclass
class VisemeData:
    """Data class for viseme information"""
    viseme: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    phoneme: str = ""

class ForceAlignVisemeExtractor:
    """
    ForceAlign-based viseme extractor that uses phoneme-level alignment
    to generate more accurate visemes synchronized with audio.
    """
    
    def __init__(self):
        """Initialize the ForceAlign viseme extractor"""
        if not FORCE_ALIGN_AVAILABLE:
            raise ImportError("ForceAlign is not available. Please install it with: pip install ForceAlign")

        self.phoneme_to_viseme_id = {
            # Silence - ID 0
            'sil': 0, 'sp': 0, '': 0,
            
            # Consonants
            'p': 1, 'b': 1, 'm': 1,           # p/b/m - ID 1
            'f': 2, 'v': 2,                   # f/v - ID 2
            'th': 3, 'dh': 3,                 # th - ID 3
            't': 4, 'd': 4, 'l': 4,           # t/d/l - ID 4
            'k': 5, 'g': 5, 'ng': 5,          # k/g/ng - ID 5
            'ch': 6, 'sh': 6, 'jh': 6, 'zh': 6,  # ch/sh/jh - ID 6
            's': 7, 'z': 7,                   # s/z - ID 7
            'n': 8,                           # n - ID 8
            'r': 9, 'w': 9, 'y': 9, 'hh': 9, # r - ID 9
            
            # Vowels
            'aa': 10, 'ae': 10, 'ah': 10,     # "a" (ah) - ID 10
            'eh': 11, 'er': 11, 'ax': 11,     # "eh" - ID 11
            'ih': 12, 'iy': 12, 'ix': 12,     # "ee" - ID 12
            'ao': 13, 'ow': 13, 'oy': 13,     # "oh" - ID 13
            'uh': 14, 'uw': 14, 'aw': 14,     # "oo" - ID 14
            'ay': 10, 'ey': 11,               # diphthongs
        }
        
        self.viseme_id_to_string = {
            0: 'sil',    # silence
            1: 'pp',     # p/b/m
            2: 'ff',     # f/v
            3: 'th',     # th
            4: 'dd',     # t/d/l
            5: 'kk',     # k/g/ng
            6: 'ch',     # ch/sh/jh
            7: 'ss',     # s/z
            8: 'nn',     # n
            9: 'rr',     # r
            10: 'aa',    # "a" (ah)
            11: 'eh',    # "eh"
            12: 'ih',    # "ee"
            13: 'oh',    # "oh"
            14: 'oo',    # "oo"
        }
        
        logger.info("ForceAlign viseme extractor initialized")
    
    async def extract_visemes(self, audio_data: np.ndarray, sample_rate: int, transcript: str) -> List[VisemeData]:
        """
        Extract visemes from audio data using ForceAlign for phoneme alignment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            transcript: Text transcript corresponding to the audio
            
        Returns:
            List of VisemeData objects with timing information
        """
        return await asyncio.to_thread(
                self.extract_visemes_sync,
                audio_data,
                sample_rate,
                transcript
            )

    
    def extract_visemes_sync(self, audio_data: np.ndarray, sample_rate: int, transcript: str) -> List[VisemeData]:
        """
        Synchronous version of viseme extraction.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            transcript: Text transcript corresponding to the audio
            
        Returns:
            List of VisemeData objects with timing information
        """
        visemes = []
        
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Write audio data to temporary file
                sf.write(temp_audio.name, audio_data, sample_rate)
                temp_audio_path = temp_audio.name
            
            try:
                # Perform forced alignment
                align = ForceAlign(temp_audio_path, transcript)
                
                words = align.inference()
                phoneme_alignments = align.phoneme_alignments
                logger.info(f"Phoneme alignments: {phoneme_alignments} ({len(phoneme_alignments)} total)")
                    # Process alignment results
                for phoneme_alignment in phoneme_alignments:
                    raw_phoneme = phoneme_alignment.phoneme
                    # remove any digits or special characters
                    raw_phoneme = ''.join(filter(str.isalpha, raw_phoneme))
                    phoneme = raw_phoneme.lower() if raw_phoneme else 'sil'
                    start_time = phoneme_alignment.time_start
                    end_time = phoneme_alignment.time_end + 0.1
                    
                    # Convert phoneme to viseme ID and then to string
                    viseme_id = self.phoneme_to_viseme_id.get(phoneme, 0)
                    viseme = self.viseme_id_to_string.get(viseme_id, 'sil')
                    
                    # Create viseme data
                    viseme_data = VisemeData(
                        viseme=viseme,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=1.0,
                        phoneme=phoneme
                    )
                    visemes.append(viseme_data)
                    
                    logger.info(f"Phoneme: {phoneme} -> Viseme: {viseme} ({start_time:.3f}s - {end_time:.3f}s)")
                
                else:
                    logger.warning("No alignment results received from ForceAlign")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error in ForceAlign viseme extraction: {e}")
            # Return default silence viseme if extraction fails
            visemes = [VisemeData(
                viseme='sil',
                start_time=0.0,
                end_time=len(audio_data) / sample_rate,
                confidence=0.5,
                phoneme='sil'
            )]
        
        logger.info(f"Extracted {len(visemes)} visemes using ForceAlign")
        return visemes
    
    def extract_visemes_from_text(self, text: str, duration: float = 1.0) -> List[VisemeData]:
        """
        Extract visemes from text without audio (fallback method).
        This creates approximate timing based on text length.
        
        Args:
            text: Text to process
            duration: Estimated duration in seconds
            
        Returns:
            List of VisemeData objects with estimated timing
        """
        visemes = []
        
        try:
            # Simple text-based viseme generation
            words = text.split()
            if not words:
                return [VisemeData(viseme='sil', start_time=0.0, end_time=duration)]
            
            time_per_word = duration / len(words)
            current_time = 0.0
            
            for word in words:
                # Simple heuristic: map common letter patterns to visemes
                viseme = self._text_to_viseme(word)
                
                viseme_data = VisemeData(
                    viseme=viseme,
                    start_time=current_time,
                    end_time=current_time + time_per_word,
                    confidence=0.5,  # Lower confidence for text-based
                    phoneme=word
                )
                visemes.append(viseme_data)
                current_time += time_per_word
        
        except Exception as e:
            logger.error(f"Error in text-based viseme extraction: {e}")
            visemes = [VisemeData(viseme='sil', start_time=0.0, end_time=duration)]
        
        return visemes
    
    def _text_to_viseme(self, word: str) -> str:
        """
        Simple text-to-viseme mapping based on common patterns.
        
        Args:
            word: Word to analyze
            
        Returns:
            Viseme string
        """
        word_lower = word.lower()
        
        # Common vowel patterns
        if any(vowel in word_lower for vowel in ['a', 'ah', 'ar']):
            return 'aa'
        elif any(vowel in word_lower for vowel in ['e', 'eh', 'er']):
            return 'ee'
        elif any(vowel in word_lower for vowel in ['i', 'ih', 'ee']):
            return 'ih'
        elif any(vowel in word_lower for vowel in ['o', 'oh', 'oo', 'ou']):
            return 'ou'
        elif any(vowel in word_lower for vowel in ['u', 'uh', 'uu']):
            return 'ou'
        
        # Common consonant patterns
        elif any(cons in word_lower for cons in ['p', 'b', 'm']):
            return 'pp'
        elif any(cons in word_lower for cons in ['f', 'v']):
            return 'ff'
        elif any(cons in word_lower for cons in ['th']):
            return 'th'
        elif any(cons in word_lower for cons in ['t', 'd', 'n']):
            return 'dd'
        elif any(cons in word_lower for cons in ['k', 'g']):
            return 'kk'
        elif any(cons in word_lower for cons in ['r', 'l']):
            return 'rr'
        elif any(cons in word_lower for cons in ['s', 'z', 'sh', 'ch', 'j']):
            return 'ss'
        
        # Default to silence for unknown patterns
        return 'sil'
    
    def get_available_visemes(self) -> List[str]:
        """
        Get list of available visemes.
        
        Returns:
            List of viseme strings
        """
        return list(set(self.phoneme_to_viseme.values()))
    
    def is_available(self) -> bool:
        """
        Check if ForceAlign is available.
        
        Returns:
            True if ForceAlign is available, False otherwise
        """
        return FORCE_ALIGN_AVAILABLE
