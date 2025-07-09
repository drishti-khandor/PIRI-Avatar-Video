"""
Avatar-related data models and types
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class VisemeType(str, Enum):
    """Supported viseme types"""
    SIL = "sil"
    AA = "AA"
    AE = "AE"
    AH = "AH"
    AO = "AO"
    AW = "AW"
    AY = "AY"
    CH = "CH"
    DD = "DD"
    EH = "EH"
    ER = "ER"
    EY = "EY"
    FF = "FF"
    HH = "HH"
    IH = "IH"
    IY = "IY"
    JH = "JH"
    KK = "KK"
    LL = "LL"
    MM = "MM"
    NN = "NN"
    NG = "NG"
    OW = "OW"
    OY = "OY"
    PP = "PP"
    RR = "RR"
    SS = "SS"
    SH = "SH"
    TH = "TH"
    TT = "TT"
    UH = "UH"
    UW = "UW"
    VV = "VV"
    WW = "WW"
    YY = "YY"
    ZZ = "ZZ"


class EmotionType(str, Enum):
    """Supported emotion types"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"


class BlendShape(BaseModel):
    """Individual blend shape definition"""
    name: str
    value: float = Field(ge=0.0, le=1.0)
    
    @validator('value')
    def clamp_value(cls, v):
        """Ensure blend shape value is between 0 and 1"""
        return max(0.0, min(1.0, v))


class VisemeData(BaseModel):
    """Viseme data with timing information"""
    viseme: str
    phoneme: Optional[str] = None
    start_time: float
    end_time: float
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    blend_shapes: Dict[str, float] = Field(default_factory=dict)
    emotion: Optional[EmotionType] = EmotionType.NEUTRAL


class VisemeRequest(BaseModel):
    """Request to trigger a specific viseme"""
    phoneme: str
    duration: Optional[float] = Field(gt=0, default=0.1)
    emotion: Optional[EmotionType] = EmotionType.NEUTRAL


class VisemeSequence(BaseModel):
    """Sequence of visemes for animation"""
    items: List[VisemeData]
    audio_duration: float
    emotion: Optional[EmotionType] = EmotionType.NEUTRAL


class AvatarState(BaseModel):
    """Current avatar state"""
    current_viseme: str = "sil"
    current_emotion: EmotionType = EmotionType.NEUTRAL
    blend_shapes: Dict[str, float] = Field(default_factory=dict)
    is_speaking: bool = False
    connected_clients: int = 0


class AvatarUpdate(BaseModel):
    """Avatar update message for WebSocket"""
    type: str = "avatar_update"
    blend_shapes: Dict[str, float]
    viseme: Optional[str] = None
    emotion: Optional[EmotionType] = None
    timestamp: float
