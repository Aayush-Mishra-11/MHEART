"""
MHEART Configuration Module
Handles all configuration settings for the application.
"""
import os
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    user: str = "mheart"
    password: str = "mheart_password"
    database: str = "mheart_db"


class RedisConfig(BaseModel):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class OllamaConfig(BaseModel):
    """Ollama LLM configuration"""
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.7
    max_tokens: int = 500


class WhisperConfig(BaseModel):
    """Whisper STT configuration"""
    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "float16"


class CrisisKeywordsConfig(BaseModel):
    """Crisis keyword weights"""
    high_risk: List[str] = [
        "suicide", "kill myself", "want to die", "end it all", "overdose",
        "hang myself", "shoot myself", "jump off", "slit my wrists"
    ]
    medium_risk: List[str] = [
        "hurt myself", "self-harm", "cutting", "better off dead",
        "nothing to live for", "no point in living", "kill me"
    ]
    low_risk: List[str] = [
        "tired of living", "no reason to live", "want to disappear",
        "wish I was dead", "better without me"
    ]


class EmotionConfig(BaseModel):
    """Emotion detection configuration"""
    positive: List[str] = ["happy", "surprise", "joy"]
    negative: List[str] = ["sad", "angry", "fear", "disgust", "anger"]
    neutral: List[str] = ["neutral"]


class FusionWeightsConfig(BaseModel):
    """Multimodal fusion weights"""
    text: float = 0.2
    audio: float = 0.3
    video: float = 0.5


class CrisisWeightsConfig(BaseModel):
    """Crisis detection weights"""
    text: float = 0.2
    audio: float = 0.3
    video: float = 0.5
    thresholds: Dict[str, float] = {
        "text": 0.8,
        "audio": 0.7,
        "video": 0.6
    }


class Settings(BaseSettings):
    """Main settings class"""
    # App settings
    app_name: str = "MHEART"
    app_version: str = "1.0.0"
    debug: bool = True
    secret_key: str = "mheart-secret-key-change-in-production"

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    knowledge_base_path: Path = data_dir / "knowledge_base" / "guidelines.txt"
    vector_db_path: Path = data_dir / "vector_db"

    # Database
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()

    # LLM
    ollama: OllamaConfig = OllamaConfig()

    # Models
    whisper: WhisperConfig = WhisperConfig()

    # Crisis detection
    crisis_keywords: CrisisKeywordsConfig = CrisisKeywordsConfig()
    emotion: EmotionConfig = EmotionConfig()
    fusion_weights: FusionWeightsConfig = FusionWeightsConfig()
    crisis_weights: CrisisWeightsConfig = CrisisWeightsConfig()

    # Processing
    max_audio_duration: int = 60  # seconds
    max_video_frames: int = 30
    batch_size: int = 8

    # Crisis hotlines
    crisis_hotlines: Dict[str, str] = {
        "US": "988 (Suicide & Crisis Lifeline)",
        "UK": "116 123 (Samaritans)",
        "Canada": "988 (Talk Suicide Canada)",
        "Australia": "13 11 14 (Lifeline)"
    }

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# Global settings instance
settings = Settings()
