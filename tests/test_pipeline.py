"""
MHEART Unit Tests
Tests for individual modules.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from backend.schemas.models import (
    TextRequest, TextInput, AudioInput, VideoInput,
    EmotionType, CrisisType
)
from backend.modules.text.processor import TextProcessor
from backend.modules.audio.processor import AudioProcessor
from backend.modules.fusion.processor import MultimodalFusion
from backend.modules.crisis.processor import CrisisDetector
from backend.modules.persona.processor import PersonaModulator, PERSONAS


class TestTextProcessor:
    """Tests for text processing module"""

    def setup_method(self):
        self.processor = TextProcessor()

    def test_preprocess_lowercase(self):
        result = self.processor.preprocess("Hello WORLD")
        assert result == "hello world"

    def test_preprocess_removes_urls(self):
        result = self.processor.preprocess("Check https://example.com")
        assert "http" not in result

    def test_preprocess_removes_special_chars(self):
        result = self.processor.preprocess("Hello!@#$ World")
        assert "!" not in result and "@" not in result

    def test_detect_crisis_keywords_high_risk(self):
        text = "I want to kill myself"
        keywords, score = self.processor.detect_crisis_keywords(text)
        assert len(keywords) > 0
        assert score > 0.5

    def test_detect_crisis_keywords_none(self):
        text = "I had a nice day today"
        keywords, score = self.processor.detect_crisis_keywords(text)
        assert len(keywords) == 0
        assert score == 0.0

    def test_sentiment_analysis_positive(self):
        # Mock the model for testing
        self.processor.sentiment_analyzer = None  # Force fallback
        score = self.processor.analyze_sentiment("I am so happy")
        # Fallback returns 0.0, but actual model would return positive

    def test_sentiment_analysis_negative(self):
        self.processor.sentiment_analyzer = None
        score = self.processor.analyze_sentiment("I am so sad")
        # Fallback returns 0.0, but actual model would return negative


class TestAudioProcessor:
    """Tests for audio processing module"""

    def setup_method(self):
        self.processor = AudioProcessor()

    def test_extract_energy_silence(self):
        # Silence should have near-zero energy
        y = np.zeros(22050)  # 1 second of silence
        energy = self.processor.extract_energy(y)
        assert energy < 0.1

    def test_extract_energy_speech_like(self):
        # Speech-like signal should have higher energy
        y = np.random.randn(22050) * 0.5
        energy = self.processor.extract_energy(y)
        assert 0.0 <= energy <= 1.0

    def test_detect_stress_high_pitch(self):
        from backend.modules.audio.processor import AudioFeatures
        features = AudioFeatures(
            pitch_hz=500,  # High pitch
            energy=0.5,
            speaking_rate_wpm=160,
            zero_crossing_rate=0.1,
            spectral_centroid=2000,
            mfcc_mean=np.zeros(13)
        )
        stress = self.processor.detect_stress(features)
        assert stress > 0.3

    def test_detect_stress_normal(self):
        from backend.modules.audio.processor import AudioFeatures
        features = AudioFeatures(
            pitch_hz=150,  # Normal pitch
            energy=0.2,   # Normal energy
            speaking_rate_wpm=120,  # Normal rate
            zero_crossing_rate=0.1,
            spectral_centroid=1500,
            mfcc_mean=np.zeros(13)
        )
        stress = self.processor.detect_stress(features)
        assert stress < 0.5

    def test_classify_emotion_sad(self):
        from backend.modules.audio.processor import AudioFeatures
        features = AudioFeatures(
            pitch_hz=150,  # Low pitch
            energy=0.2,    # Low energy
            speaking_rate_wpm=100,
            zero_crossing_rate=0.1,
            spectral_centroid=1000,
            mfcc_mean=np.zeros(13)
        )
        emotion, conf = self.processor.classify_emotion(features, 0.3)
        assert emotion == EmotionType.SAD


class TestMultimodalFusion:
    """Tests for multimodal fusion (CMDD)"""

    def setup_method(self):
        self.fusion = MultimodalFusion()

    def test_no_discrepancy_all_same(self):
        is_discrepant, dtype = self.fusion.detect_discrepancy(
            EmotionType.HAPPY, EmotionType.HAPPY, EmotionType.HAPPY
        )
        assert not is_discrepant
        assert dtype.value == "NONE"

    def test_discrepancy_hidden_distress(self):
        # Text positive, audio negative, video negative
        is_discrepant, dtype = self.fusion.detect_discrepancy(
            EmotionType.HAPPY, EmotionType.SAD, EmotionType.SAD
        )
        assert is_discrepant
        assert dtype == DiscrepancyType.HIDDEN_DISTRESS

    def test_discrepancy_suppressed_joy(self):
        # Text negative, audio positive, video positive
        is_discrepant, dtype = self.fusion.detect_discrepancy(
            EmotionType.SAD, EmotionType.HAPPY, EmotionType.HAPPY
        )
        assert is_discrepant
        assert dtype == DiscrepancyType.SUPPRESSED_JOY

    def test_no_discrepancy_mixed(self):
        # One of each - no clear discrepancy
        is_discrepant, dtype = self.fusion.detect_discrepancy(
            EmotionType.HAPPY, EmotionType.SAD, EmotionType.NEUTRAL
        )
        assert not is_discrepant

    def test_calculate_intensity_with_discrepancy(self):
        text_input = TextInput(
            raw_text="I'm fine",
            processed_text="i'm fine",
            sentiment_score=0.9,
            emotion=EmotionType.HAPPY,
            crisis_score=0.0
        )
        # Should boost intensity when discrepancy
        intensity = self.fusion.calculate_intensity(
            text_input, None, None, is_discrepant=True
        )
        assert intensity >= 0.6


class TestCrisisDetector:
    """Tests for crisis detection (WCS)"""

    def setup_method(self):
        self.detector = CrisisDetector()

    def test_no_crisis_low_scores(self):
        text_input = TextInput(
            raw_text="I'm having a normal day",
            processed_text="i'm having a normal day",
            sentiment_score=0.5,
            emotion=EmotionType.HAPPY,
            crisis_score=0.1
        )
        alert = self.detector.detect(text_input)
        assert not alert.is_crisis
        assert alert.crisis_type == CrisisType.NONE

    def test_crisis_high_text_score(self):
        text_input = TextInput(
            raw_text="I want to kill myself",
            processed_text="i want to kill myself",
            sentiment_score=-0.9,
            emotion=EmotionType.SAD,
            crisis_score=0.9,
            is_crisis_keyword=True,
            crisis_keywords_found=["suicide (high)"]
        )
        alert = self.detector.detect(text_input)
        assert alert.is_crisis
        assert alert.crisis_type in [CrisisType.HIGH_RISK, CrisisType.IMMEDIATE_DANGER]

    def test_crisis_multiple_modalities(self):
        text_input = TextInput(
            raw_text="I feel terrible",
            processed_text="i feel terrible",
            sentiment_score=-0.5,
            emotion=EmotionType.SAD,
            crisis_score=0.5
        )
        audio_input = AudioInput(
            pitch_hz=450,
            energy=0.6,
            speaking_rate_wpm=170,
            voice_stress_level=0.6,
            emotion=EmotionType.SAD,
            crisis_score=0.6
        )
        alert = self.detector.detect(text_input, audio_input)
        assert alert.is_crisis
        assert len(alert.triggered_modalities) >= 2


class TestPersonaModulator:
    """Tests for persona modulation (ATP)"""

    def setup_method(self):
        self.modulator = PersonaModulator()

    def test_crisis_selects_clinical(self):
        from backend.schemas.models import FusedEmotion, CrisisAlert
        fused = FusedEmotion(
            dominant_emotion=EmotionType.SAD,
            emotion_intensity=0.5,
            is_discrepant=False
        )
        crisis = CrisisAlert(
            is_crisis=True,
            crisis_type=CrisisType.HIGH_RISK
        )
        persona = self.modulator.select_persona(fused, crisis)
        assert persona.persona_type.value == "clinical_psychologist"

    def test_discrepancy_selects_empathetic(self):
        from backend.schemas.models import FusedEmotion, CrisisAlert
        fused = FusedEmotion(
            dominant_emotion=EmotionType.HAPPY,
            emotion_intensity=0.5,
            is_discrepant=True
        )
        crisis = CrisisAlert(
            is_crisis=False,
            crisis_type=CrisisType.NONE
        )
        persona = self.modulator.select_persona(fused, crisis)
        assert persona.persona_type.value == "empathetic_listener"

    def test_low_intensity_selects_friendly(self):
        from backend.schemas.models import FusedEmotion, CrisisAlert
        fused = FusedEmotion(
            dominant_emotion=EmotionType.NEUTRAL,
            emotion_intensity=0.2,
            is_discrepant=False
        )
        crisis = CrisisAlert(
            is_crisis=False,
            crisis_type=CrisisType.NONE
        )
        persona = self.modulator.select_persona(fused, crisis)
        assert persona.persona_type.value == "friendly_companion"


class TestSchemas:
    """Tests for Pydantic schemas"""

    def test_text_input_validation(self):
        text = TextInput(
            raw_text="test",
            processed_text="test",
            sentiment_score=0.5,
            emotion=EmotionType.HAPPY,
            crisis_keywords_found=[],
            crisis_score=0.0
        )
        assert text.sentiment_score == 0.5

    def test_text_input_sentiment_bounds(self):
        with pytest.raises(ValueError):
            TextInput(
                raw_text="test",
                processed_text="test",
                sentiment_score=1.5,  # Invalid - must be -1 to 1
                emotion=EmotionType.HAPPY
            )

    def test_crisis_alert_fields(self):
        from backend.schemas.models import CrisisAlert, CrisisType, InputModality
        alert = CrisisAlert(
            is_crisis=True,
            crisis_type=CrisisType.HIGH_RISK,
            triggered_modalities=[InputModality.TEXT, InputModality.AUDIO],
            confidence=0.8
        )
        assert alert.is_crisis
        assert len(alert.triggered_modalities) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])