"""
AI/LLM client for chat interactions with emotion detection
"""
import logging
from typing import Optional, Dict, List, Tuple
from openai import AzureOpenAI
import re

from app.config.settings import settings
from app.models.avatar import EmotionType


logger = logging.getLogger(__name__)


class LLMClient:
    """Handles LLM interactions with Azure OpenAI"""
    
    def __init__(self):
        self.client = None
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = """You are a helpful AI assistant with a 3D VRM avatar. 
        Keep responses concise and natural for speech synthesis. 
        Express emotions appropriately through your responses."""
        
        self._initialize_client()
        self._reset_conversation()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        if all([
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_deployment_name
        ]):
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                )
                logger.info("Azure OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("Missing Azure OpenAI configuration")
    
    def _reset_conversation(self):
        """Reset conversation history"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def detect_emotion(self, text: str) -> EmotionType:
        """Detect emotion from text using simple heuristics"""
        text_lower = text.lower()
        
        # Emotion keywords mapping
        emotion_patterns = {
            EmotionType.HAPPY: [
                r'\b(happy|joy|excited|wonderful|great|amazing|fantastic|love|laugh|smile)\b',
                r'[😊😄😃😀🙂😁]'
            ],
            EmotionType.SAD: [
                r'\b(sad|sorry|depressed|unhappy|cry|tears|miss|lonely)\b',
                r'[😢😭😞😔]'
            ],
            EmotionType.ANGRY: [
                r'\b(angry|mad|furious|annoyed|hate|stupid|damn)\b',
                r'[😠😡🤬]'
            ],
            EmotionType.SURPRISED: [
                r'\b(wow|amazing|surprised|shock|unexpected|incredible)\b',
                r'[😮😲😱]'
            ],
            EmotionType.FEARFUL: [
                r'\b(afraid|scared|fear|terrified|nervous|worry)\b',
                r'[😨😰😱]'
            ]
        }
        
        # Check each emotion pattern
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return emotion
        
        # Default to neutral
        return EmotionType.NEUTRAL
    
    async def generate_response(
        self, 
        user_input: str,
        max_tokens: int = 150
    ) -> Tuple[str, EmotionType]:
        """Generate AI response with emotion detection"""
        if not self.client:
            return "I'm sorry, I'm not properly configured to respond.", EmotionType.NEUTRAL
        
        try:
            # Add user message
            self.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract response text
            ai_response = response.choices[0].message.content
            
            # Add to conversation history
            self.messages.append({"role": "assistant", "content": ai_response})
            
            # Detect emotion
            emotion = self.detect_emotion(ai_response)
            
            # Limit conversation history to prevent token overflow
            if len(self.messages) > 20:
                self.messages = [self.messages[0]] + self.messages[-10:]
            
            return ai_response, emotion
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I encountered an error. Please try again.", EmotionType.NEUTRAL
    
    def generate_response_sync(
        self, 
        user_input: str,
        max_tokens: int = 150
    ) -> Tuple[str, EmotionType]:
        """Synchronous version of generate_response for use in non-async contexts"""
        if not self.client:
            return "I'm sorry, I'm not properly configured to respond.", EmotionType.NEUTRAL
        
        try:
            # Add user message
            self.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract response text
            ai_response = response.choices[0].message.content
            
            # Add to conversation history
            self.messages.append({"role": "assistant", "content": ai_response})
            
            # Detect emotion
            emotion = self.detect_emotion(ai_response)
            
            # Limit conversation history to prevent token overflow
            if len(self.messages) > 20:
                self.messages = [self.messages[0]] + self.messages[-10:]
            
            return ai_response, emotion
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I encountered an error. Please try again.", EmotionType.NEUTRAL
    
    def add_context(self, context: str):
        """Add context to the system prompt"""
        self.system_prompt = f"{self.system_prompt}\n\nAdditional context: {context}"
        self._reset_conversation()
