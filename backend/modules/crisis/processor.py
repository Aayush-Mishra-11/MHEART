"""
MHEART Crisis Detection Module (WCS - Weighted Crisis Scoring)
THIS IS THE SECOND NOVEL CONTRIBUTION

Calculates weighted crisis score from all modalities.
Determines crisis severity and escalation requirements.

Algorithm:
1. Get crisis scores from all modalities
2. Apply thresholds to each modality
3. Count crisis modalities
4. Calculate weighted crisis score
5. Determine crisis status and type
"""

import time
from typing import List

from backend.schemas.models import (
    TextInput, AudioInput, VideoInput, CrisisAlert, CrisisType, InputModality
)
from backend.core.config import settings


class CrisisDetector:
    """
    Weighted Crisis Scoring (WCS) module.
    Combines crisis indicators from all modalities.
    """

    def __init__(self):
        self.weights = settings.crisis_weights
        self.thresholds = settings.crisis_weights.thresholds
        self.hotlines = settings.crisis_hotlines

    def _get_crisis_modalities(
        self,
        text_input: TextInput = None,
        audio_input: AudioInput = None,
        video_input: VideoInput = None
    ) -> List[InputModality]:
        """Count which modalities are in crisis state"""
        triggered = []

        if text_input and text_input.crisis_score > self.thresholds["text"]:
            triggered.append(InputModality.TEXT)

        if audio_input and audio_input.crisis_score > self.thresholds["audio"]:
            triggered.append(InputModality.AUDIO)

        if video_input and video_input.crisis_score > self.thresholds["video"]:
            triggered.append(InputModality.VIDEO)

        return triggered

    def _calculate_weighted_score(
        self,
        text_input: TextInput = None,
        audio_input: AudioInput = None,
        video_input: VideoInput = None
    ) -> float:
        """Calculate weighted crisis score"""
        text_score = text_input.crisis_score if text_input else 0.0
        audio_score = audio_input.crisis_score if audio_input else 0.0
        video_score = video_input.crisis_score if video_input else 0.0

        # Count active modalities
        active_count = sum([
            text_input is not None,
            audio_input is not None,
            video_input is not None
        ])

        if active_count == 0:
            return 0.0

        # Normalize weights based on active modalities
        if active_count == 1:
            # Only one modality - use it at full weight
            if text_input is not None:
                return min(text_score, 1.0)
            elif audio_input is not None:
                return min(audio_score, 1.0)
            else:
                return min(video_score, 1.0)
        elif active_count == 2:
            # Two modalities - redistribute weights proportionally
            if text_input is None:
                # Audio + Video: 0.3 + 0.5 = 0.8, so normalize to 1.0
                audio_weight = 0.3 / 0.8
                video_weight = 0.5 / 0.8
                weighted = audio_score * audio_weight + video_score * video_weight
            elif audio_input is None:
                # Text + Video: 0.2 + 0.5 = 0.7, so normalize to 1.0
                text_weight = 0.2 / 0.7
                video_weight = 0.5 / 0.7
                weighted = text_score * text_weight + video_score * video_weight
            else:
                # Text + Audio: 0.2 + 0.3 = 0.5, so normalize to 1.0
                text_weight = 0.2 / 0.5
                audio_weight = 0.3 / 0.5
                weighted = text_score * text_weight + audio_score * audio_weight
        else:
            # All three modalities - use full weights
            weighted = (
                text_score * self.weights.text +
                audio_score * self.weights.audio +
                video_score * self.weights.video
            )

        return min(weighted, 1.0)

    def _determine_crisis_type(
        self,
        weighted_score: float,
        triggered_count: int,
        text_input: TextInput = None
    ) -> CrisisType:
        """Determine the crisis type based on score and indicators"""
        # Check for immediate danger keywords
        if text_input and text_input.is_crisis_keyword:
            high_risk_keywords = any(
                "high" in kw.lower() for kw in text_input.crisis_keywords_found
            )
            if high_risk_keywords and weighted_score > 0.7:
                return CrisisType.IMMEDIATE_DANGER

        # Immediate danger threshold
        if weighted_score > 0.85:
            return CrisisType.IMMEDIATE_DANGER

        # High risk
        if weighted_score > 0.65 or triggered_count >= 2:
            return CrisisType.HIGH_RISK

        # Moderate concern
        if weighted_score > 0.4 or triggered_count >= 1:
            return CrisisType.MODERATE_CONCERN

        # Low risk (some indicators present)
        if weighted_score > 0.2:
            return CrisisType.LOW_RISK

        return CrisisType.NONE

    def _generate_recommendation(self, crisis_type: CrisisType, triggered: List[InputModality]) -> str:
        """Generate human-readable recommendation"""
        if crisis_type == CrisisType.IMMEDIATE_DANGER:
            return (
                "URGENT: Immediate safety concern detected. "
                "Please reach out to crisis services immediately. "
                "You deserve support right now."
            )
        elif crisis_type == CrisisType.HIGH_RISK:
            return (
                "High risk indicators detected. "
                "Please consider reaching out to a mental health professional or crisis line. "
                "Your feelings are valid and help is available."
            )
        elif crisis_type == CrisisType.MODERATE_CONCERN:
            return (
                "Some concerning indicators detected. "
                "If you feel overwhelmed, please don't hesitate to talk to someone. "
                "Professional support can help."
            )
        elif crisis_type == CrisisType.LOW_RISK:
            return (
                "Minor indicators present. "
                "Continue to monitor your wellbeing. "
                "Self-care activities may help."
            )
        return ""

    def detect(
        self,
        text_input: TextInput = None,
        audio_input: AudioInput = None,
        video_input: VideoInput = None
    ) -> CrisisAlert:
        """
        Detect crisis from all available modalities.
        Returns CrisisAlert with severity and recommendations.
        """
        start_time = time.time()

        # Get crisis modalities
        triggered_modalities = self._get_crisis_modalities(text_input, audio_input, video_input)

        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(text_input, audio_input, video_input)

        # Determine crisis type
        crisis_type = self._determine_crisis_type(
            weighted_score,
            len(triggered_modalities),
            text_input
        )

        # Determine if crisis
        is_crisis = (
            len(triggered_modalities) >= 2 or
            weighted_score > 0.6 or
            crisis_type in [CrisisType.IMMEDIATE_DANGER, CrisisType.HIGH_RISK]
        )

        # Determine if escalation required
        escalation_required = (
            crisis_type == CrisisType.IMMEDIATE_DANGER or
            weighted_score > 0.8
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(crisis_type, triggered_modalities)

        processing_time = (time.time() - start_time) * 1000

        return CrisisAlert(
            is_crisis=is_crisis,
            confidence=weighted_score,
            triggered_modalities=triggered_modalities,
            escalation_required=escalation_required,
            crisis_type=crisis_type,
            weighted_score=weighted_score,
            recommendation=recommendation,
            hotlines=self.hotlines,
            processing_time_ms=processing_time
        )


# Global instance
crisis_detector = CrisisDetector()
