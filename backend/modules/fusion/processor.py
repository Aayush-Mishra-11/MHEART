"""
MHEART Multimodal Fusion Module (CMDD - Cross-Modal Discrepancy Detection)
THIS IS THE FIRST NOVEL CONTRIBUTION

Detects when different modalities report conflicting emotions, indicating
the user may be hiding their true feelings.

Algorithm:
1. Collect emotions from all modalities
2. Categorize into positive/negative/neutral
3. Detect discrepancy when positive/negative differ by 2+
4. Calculate weighted emotion intensity
5. Determine dominant emotion
"""

import time
from typing import Dict, List, Tuple

from backend.schemas.models import (
    TextInput, AudioInput, VideoInput, FusedEmotion,
    EmotionType, DiscrepancyType, InputModality
)
from backend.core.config import settings


class MultimodalFusion:
    """
    Cross-Modal Discrepancy Detection (CMDD) module.
    Detects hidden emotions by comparing emotions across modalities.
    """

    def __init__(self):
        self.positive_emotions = set(settings.emotion.positive)
        self.negative_emotions = set(settings.emotion.negative)
        self.neutral_emotions = set(settings.emotion.neutral)
        self.weights = settings.fusion_weights

    def _categorize_emotion(self, emotion: EmotionType) -> str:
        """Categorize emotion as positive, negative, or neutral"""
        if emotion in [EmotionType.HAPPY, EmotionType.SURPRISE, EmotionType.JOY]:
            return "positive"
        elif emotion in [EmotionType.SAD, EmotionType.ANGRY, EmotionType.FEAR, EmotionType.DISGUST]:
            return "negative"
        else:
            return "neutral"

    def _count_categories(self, emotions: List[EmotionType]) -> Dict[str, int]:
        """Count how many modalities report each category"""
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for emo in emotions:
            category = self._categorize_emotion(emo)
            counts[category] += 1
        return counts

    def detect_discrepancy(
        self,
        text_emotion: EmotionType,
        audio_emotion: EmotionType,
        video_emotion: EmotionType,
        has_text: bool = True,
        has_audio: bool = True,
        has_video: bool = True
    ) -> Tuple[bool, DiscrepancyType]:
        """
        Detect cross-modal discrepancy.
        Returns (is_discrepant, discrepancy_type)

        Discrepancy is detected when positive and negative counts differ by 2+.
        Only counts modalities that are actually present (has_* flags).
        """
        # Only detect discrepancy when we have at least 2 modalities
        active_modalities = sum([has_text, has_audio, has_video])
        if active_modalities < 2:
            return False, DiscrepancyType.NONE

        # Build list of only present emotions
        emotions = []
        if has_text:
            emotions.append(text_emotion)
        if has_audio:
            emotions.append(audio_emotion)
        if has_video:
            emotions.append(video_emotion)

        counts = self._count_categories(emotions)

        positive_count = counts["positive"]
        negative_count = counts["negative"]

        # Check if discrepancy exists
        if abs(positive_count - negative_count) >= 2:
            if positive_count > negative_count:
                return True, DiscrepancyType.HIDDEN_DISTRESS
            else:
                return True, DiscrepancyType.SUPPRESSED_JOY

        return False, DiscrepancyType.NONE

    def calculate_intensity(
        self,
        text_input: TextInput,
        audio_input: AudioInput,
        video_input: VideoInput,
        is_discrepant: bool
    ) -> float:
        """
        Calculate weighted emotion intensity.
        Video weighted highest because micro-expressions are hard to fake.
        """
        # Base intensity from sentiment
        text_intensity = abs(text_input.sentiment_score) if text_input else 0.0

        # Audio stress
        audio_intensity = audio_input.voice_stress_level if audio_input else 0.0

        # Video confidence
        video_intensity = video_input.emotion_confidence if video_input else 0.0

        intensity = (
            text_intensity * self.weights.text +
            audio_intensity * self.weights.audio +
            video_intensity * self.weights.video
        )

        # If discrepancy, boost intensity
        if is_discrepant:
            intensity = max(intensity, 0.6)

        return min(intensity, 1.0)

    def determine_dominant_emotion(
        self,
        text_input: TextInput,
        audio_input: AudioInput,
        video_input: VideoInput,
        is_discrepant: bool
    ) -> Tuple[EmotionType, float]:
        """
        Determine dominant emotion using weighted voting.
        Video prioritized if discrepancy detected.
        """
        emotion_votes: Dict[EmotionType, float] = {}

        if text_input:
            text_emo = text_input.emotion
            text_weight = self.weights.text * (abs(text_input.sentiment_score) + 0.5)
            emotion_votes[text_emo] = emotion_votes.get(text_emo, 0) + text_weight

        if audio_input:
            audio_emo = audio_input.emotion
            audio_weight = self.weights.audio * audio_input.emotion_confidence
            emotion_votes[audio_emo] = emotion_votes.get(audio_emo, 0) + audio_weight

        if video_input:
            video_emo = video_input.emotion
            # Video gets higher weight, especially if discrepancy
            video_weight = self.weights.video * video_input.emotion_confidence
            if is_discrepant:
                video_weight *= 1.5  # Boost video importance when discrepancy detected
            emotion_votes[video_emo] = emotion_votes.get(video_emo, 0) + video_weight

        if not emotion_votes:
            return EmotionType.NEUTRAL, 0.0

        # Get dominant emotion
        dominant = max(emotion_votes.items(), key=lambda x: x[1])
        confidence = dominant[1] / sum(emotion_votes.values()) if sum(emotion_votes.values()) > 0 else 0.0

        return dominant[0], min(confidence, 1.0)

    def create_emotion_agreement(
        self,
        text_input: TextInput,
        audio_input: AudioInput,
        video_input: VideoInput
    ) -> Dict[InputModality, EmotionType]:
        """Create mapping of modality to detected emotion"""
        agreement = {}

        if text_input:
            agreement[InputModality.TEXT] = text_input.emotion
        if audio_input:
            agreement[InputModality.AUDIO] = audio_input.emotion
        if video_input:
            agreement[InputModality.VIDEO] = video_input.emotion

        return agreement

    def fuse(
        self,
        text_input: TextInput = None,
        audio_input: AudioInput = None,
        video_input: VideoInput = None
    ) -> FusedEmotion:
        """
        Fuse emotions from all available modalities.
        Returns FusedEmotion with discrepancy detection.
        """
        start_time = time.time()

        # Track which modalities are actually present
        has_text = text_input is not None
        has_audio = audio_input is not None
        has_video = video_input is not None

        # Get emotions from each modality (default to NEUTRAL if not present)
        text_emotion = text_input.emotion if text_input else EmotionType.NEUTRAL
        audio_emotion = audio_input.emotion if audio_input else EmotionType.NEUTRAL
        video_emotion = video_input.emotion if video_input else EmotionType.NEUTRAL

        # Detect discrepancy (only if we have enough modalities)
        is_discrepant, discrepancy_type = self.detect_discrepancy(
            text_emotion, audio_emotion, video_emotion,
            has_text, has_audio, has_video
        )

        # Calculate intensity
        intensity = self.calculate_intensity(text_input, audio_input, video_input, is_discrepant)

        # Determine dominant emotion
        dominant_emotion, confidence = self.determine_dominant_emotion(
            text_input, audio_input, video_input, is_discrepant
        )

        # Create agreement mapping
        agreement = self.create_emotion_agreement(text_input, audio_input, video_input)

        processing_time = (time.time() - start_time) * 1000

        return FusedEmotion(
            dominant_emotion=dominant_emotion,
            emotion_intensity=intensity,
            is_discrepant=is_discrepant,
            discrepancy_type=discrepancy_type,
            emotion_agreement=agreement,
            confidence=confidence,
            fusion_method="CMDD_weighted_voting",
            processing_time_ms=processing_time
        )


# Global instance
multimodal_fusion = MultimodalFusion()
