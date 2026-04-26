"""
MHEART Pydantic Schemas
Defines all data models for the application.
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class EmotionType(str, Enum):
    """Emotion types enum"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    JOY = "joy"


class PersonaType(str, Enum):
    """Persona types enum"""
    CLINICAL_PSYCHOLOGIST = "clinical_psychologist"
    EMPATHETIC_LISTENER = "empathetic_listener"
    FRIENDLY_COMPANION = "friendly_companion"


class CrisisType(str, Enum):
    """Crisis severity types"""
    IMMEDIATE_DANGER = "IMMEDIATE_DANGER"
    HIGH_RISK = "HIGH_RISK"
    MODERATE_CONCERN = "MODERATE_CONCERN"
    LOW_RISK = "LOW_RISK"
    NONE = "NONE"


class DiscrepancyType(str, Enum):
    """Cross-modal discrepancy types"""
    HIDDEN_DISTRESS = "HIDDEN_DISTRESS"
    SUPPRESSED_JOY = "SUPPRESSED_JOY"
    NONE = "NONE"


class InputModality(str, Enum):
    """Input modality types"""
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


# ============================================================================
# TEXT PROCESSING SCHEMAS
# ============================================================================

class TextInput(BaseModel):
    """Text input processing result"""
    raw_text: str
    processed_text: str
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Sentiment from -1 (negative) to +1 (positive)")
    emotion: EmotionType
    emotion_scores: Dict[EmotionType, float] = Field(default_factory=dict)
    crisis_keywords_found: List[str] = Field(default_factory=list)
    crisis_score: float = Field(ge=0.0, le=1.0, description="Crisis score 0-1")
    is_crisis_keyword: bool = False
    processing_time_ms: float = 0.0


class TextRequest(BaseModel):
    """Request model for text input"""
    text: str
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None


# ============================================================================
# AUDIO PROCESSING SCHEMAS
# ============================================================================

class AudioInput(BaseModel):
    """Audio input processing result"""
    pitch_hz: float = 0.0
    energy: float = 0.0  # 0-1
    speaking_rate_wpm: float = 0.0
    voice_stress_level: float = Field(ge=0.0, le=1.0)
    emotion: EmotionType
    emotion_confidence: float = Field(ge=0.0, le=1.0)
    crisis_score: float = Field(ge=0.0, le=1.0)
    transcript: Optional[str] = None
    processing_time_ms: float = 0.0
    voice_detected: bool = True


class AudioRequest(BaseModel):
    """Request model for audio input"""
    audio_data: bytes
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    format: str = "wav"


# ============================================================================
# VIDEO PROCESSING SCHEMAS
# ============================================================================

class VideoInput(BaseModel):
    """Video input processing result"""
    face_detected: bool = False
    landmark_points: List[List[float]] = Field(default_factory=list)
    emotion: EmotionType
    emotion_confidence: float = Field(ge=0.0, le=1.0)
    all_emotions: Dict[EmotionType, float] = Field(default_factory=dict)
    crisis_score: float = Field(ge=0.0, le=1.0)
    eye_contact_ratio: float = 0.0  # 0-1
    facial_activity: float = 0.0  # 0-1
    processing_time_ms: float = 0.0


class VideoRequest(BaseModel):
    """Request model for video input"""
    frame_data: bytes  # Base64 encoded image
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None


# ============================================================================
# MULTIMODAL FUSION SCHEMAS (CMDD - Cross-Modal Discrepancy Detection)
# ============================================================================

class FusedEmotion(BaseModel):
    """Result of multimodal fusion"""
    dominant_emotion: EmotionType
    emotion_intensity: float = Field(ge=0.0, le=1.0)
    is_discrepant: bool = False
    discrepancy_type: DiscrepancyType = DiscrepancyType.NONE
    emotion_agreement: Dict[InputModality, EmotionType] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    fusion_method: str = "weighted_voting"
    processing_time_ms: float = 0.0


# ============================================================================
# CRISIS DETECTION SCHEMAS (WCS - Weighted Crisis Scoring)
# ============================================================================

class CrisisAlert(BaseModel):
    """Crisis detection result"""
    is_crisis: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    triggered_modalities: List[InputModality] = Field(default_factory=list)
    escalation_required: bool = False
    crisis_type: CrisisType = CrisisType.NONE
    weighted_score: float = 0.0
    recommendation: str = ""
    hotlines: Dict[str, str] = Field(default_factory=dict)
    processing_time_ms: float = 0.0


# ============================================================================
# PERSONA MODULATION SCHEMAS (ATP - Adaptive Therapeutic Persona)
# ============================================================================

class PersonaConfig(BaseModel):
    """Persona configuration"""
    persona_type: PersonaType
    system_prompt: str
    tone: str
    response_length: str  # "short", "medium", "long"
    max_response_length: int = 500


# ============================================================================
# RAG SCHEMAS
# ============================================================================

class RetrievedContext(BaseModel):
    """Retrieved context from knowledge base"""
    content: str
    source: str
    relevance_score: float


# ============================================================================
# LLM RESPONSE SCHEMAS
# ============================================================================

class LLMResponse(BaseModel):
    """LLM generation result"""
    response_text: str
    persona_used: PersonaType
    context_used: List[RetrievedContext] = Field(default_factory=list)
    generation_time_ms: float = 0.0
    tokens_used: Optional[int] = None


# ============================================================================
# FULL PIPELINE SCHEMAS
# ============================================================================

class MultimodalInput(BaseModel):
    """Combined multimodal input"""
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    video_data: Optional[bytes] = None
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None


class PipelineResult(BaseModel):
    """Complete processing pipeline result"""
    # Individual modality results
    text_result: Optional[TextInput] = None
    audio_result: Optional[AudioInput] = None
    video_result: Optional[VideoInput] = None

    # Fusion results
    fused_emotion: FusedEmotion
    crisis_alert: CrisisAlert
    persona: PersonaConfig

    # Response
    llm_response: LLMResponse

    # Metadata
    total_processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[UUID] = None


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(BaseModel):
    """User model"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Session model"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None


class ConversationEntry(BaseModel):
    """Conversation entry model"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_message: str
    bot_response: str
    emotion_state: Dict[str, Any] = Field(default_factory=dict)
    crisis_detected: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CrisisEvent(BaseModel):
    """Crisis event model"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    crisis_score: float
    triggered_modalities: List[str] = Field(default_factory=list)
    action_taken: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# API RESPONSE SCHEMAS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None


class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str
    emotion_detected: EmotionType
    crisis_detected: bool
    crisis_type: Optional[CrisisType] = None
    persona_used: PersonaType
    processing_time_ms: float


class EmotionResponse(BaseModel):
    """Emotion state response"""
    text_emotion: Optional[EmotionType] = None
    audio_emotion: Optional[EmotionType] = None
    video_emotion: Optional[EmotionType] = None
    fused_emotion: EmotionType
    confidence: float
    is_discrepant: bool
    discrepancy_type: DiscrepancyType


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    ollama_connected: bool
