"""
MHEART Audio Processing Module
Handles all audio-based input processing including:
- Voice Activity Detection (VAD)
- Audio feature extraction (pitch, energy, speaking rate)
- Stress detection
- Audio emotion classification
"""

import time
import io
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

import librosa
import soundfile as sf

from backend.schemas.models import AudioInput, AudioRequest, EmotionType
from backend.core.config import settings


@dataclass
class AudioFeatures:
    """Extracted audio features"""
    pitch_hz: float
    energy: float
    speaking_rate_wpm: float
    zero_crossing_rate: float
    spectral_centroid: float
    mfcc_mean: np.ndarray


class AudioProcessor:
    """
    Audio processing module for mental health detection.
    Uses Librosa for feature extraction, Faster-Whisper for STT,
    and rule-based emotion classification.
    """

    def __init__(self):
        self.sample_rate = 22050
        self.vad_enabled = True
        self.whisper_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize Whisper model for speech-to-text."""
        try:
            from faster_whisper import WhisperModel
            # Use base model for good balance of speed and accuracy
            # Fall back to tiny if base unavailable
            try:
                self.whisper_model = WhisperModel(
                    "base",
                    device="auto",
                    compute_type="float16"
                )
                print("[AUDIO] Faster-Whisper (base) loaded successfully")
            except Exception:
                self.whisper_model = WhisperModel(
                    "tiny",
                    device="auto",
                    compute_type="int8"
                )
                print("[AUDIO] Faster-Whisper (tiny) loaded successfully")
        except ImportError:
            print("[AUDIO] faster-whisper not available, STT disabled")
        except Exception as e:
            print(f"[AUDIO] Failed to load Whisper model: {e}")
            self.whisper_model = None

    def transcribe_audio(self, audio_array: np.ndarray) -> Optional[str]:
        """
        Transcribe audio using Faster-Whisper.
        Returns transcript string or None if transcription fails.
        """
        if self.whisper_model is None:
            return None

        try:
            # Ensure audio is float32
            audio_float32 = audio_array.astype(np.float32)

            # Run transcription
            segments, info = self.whisper_model.transcribe(
                audio_float32,
                beam_size=5,
                language=None,  # Auto-detect
                task="transcribe"
            )

            # Combine all segments into transcript
            transcript_parts = []
            for segment in segments:
                if segment.text:
                    transcript_parts.append(segment.text.strip())

            if transcript_parts:
                return " ".join(transcript_parts)
            return None

        except Exception as e:
            print(f"[AUDIO] Transcription error: {e}")
            return None

    def load_audio(self, audio_data: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes.
        Returns audio waveform and sample rate.
        """
        try:
            # Convert bytes to audio
            audio_array, sr = librosa.load(
                io.BytesIO(audio_data),
                sr=self.sample_rate,
                mono=True
            )
            return audio_array, sr
        except Exception as e:
            print(f"[AUDIO] Error loading audio: {e}")
            # Return silence
            return np.zeros(self.sample_rate), self.sample_rate

    def extract_pitch(self, y: np.ndarray, sr: int) -> float:
        """
        Extract pitch (F0) using librosa.
        Returns pitch in Hz.
        """
        try:
            # Use pyin for fundamental frequency estimation
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=50,
                fmax=500,
                sr=sr
            )

            # Filter out unvoiced segments and get mean pitch
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0 and not np.isnan(voiced_f0).all():
                pitch = np.nanmean(voiced_f0)
                return float(pitch) if not np.isnan(pitch) else 0.0

            return 0.0

        except Exception as e:
            print(f"[AUDIO] Pitch extraction error: {e}")
            return 0.0

    def extract_energy(self, y: np.ndarray) -> float:
        """
        Extract energy (RMS) from audio.
        Returns energy value between 0 and 1.
        """
        try:
            rms = librosa.feature.rms(y=y)[0]
            # Normalize to 0-1 range
            energy = np.mean(rms)
            # Simple normalization assuming typical values
            energy = min(energy * 5, 1.0)
            return float(energy)

        except Exception as e:
            print(f"[AUDIO] Energy extraction error: {e}")
            return 0.0

    def extract_speaking_rate(self, y: np.ndarray, sr: int) -> float:
        """
        Extract speaking rate in words per minute.
        Uses onset detection to estimate speech rate.
        """
        try:
            # Detect onset frames
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                units='frames',
                backtrack=True
            )

            # Convert frames to time
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            # Calculate duration
            duration = len(y) / sr

            if duration > 0:
                # Estimate words (assuming ~2 syllables per word on average)
                # and typical English speech has ~2.5 syllables per word
                estimated_words = len(onset_times) / 2.5
                speaking_rate = (estimated_words / duration) * 60
                return float(speaking_rate)

            return 0.0

        except Exception as e:
            print(f"[AUDIO] Speaking rate extraction error: {e}")
            return 0.0

    def extract_features(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """
        Extract all audio features.
        """
        pitch = self.extract_pitch(y, sr)
        energy = self.extract_energy(y)
        speaking_rate = self.extract_speaking_rate(y, sr)

        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_rate = float(np.mean(zcr))
        except:
            zero_crossing_rate = 0.0

        try:
            spectral = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid = float(np.mean(spectral))
        except:
            spectral_centroid = 0.0

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
        except:
            mfcc_mean = np.zeros(13)

        return AudioFeatures(
            pitch_hz=pitch,
            energy=energy,
            speaking_rate_wpm=speaking_rate,
            zero_crossing_rate=zero_crossing_rate,
            spectral_centroid=spectral_centroid,
            mfcc_mean=mfcc_mean
        )

    def detect_stress(self, features: AudioFeatures) -> float:
        """
        Detect stress level from audio features.
        Returns stress score between 0 and 1.
        """
        stress_score = 0.0

        # Pitch analysis (high pitch = stress)
        if features.pitch_hz > 400:
            pitch_contribution = min((features.pitch_hz - 400) / 400, 1.0)
            stress_score += pitch_contribution * 0.4

        # Energy analysis (high energy = stress)
        if features.energy > 0.3:
            energy_contribution = min((features.energy - 0.3) / 0.7, 1.0)
            stress_score += energy_contribution * 0.3

        # Speaking rate analysis (fast speech = stress)
        if features.speaking_rate_wpm > 150:
            rate_contribution = min((features.speaking_rate_wpm - 150) / 100, 1.0)
            stress_score += rate_contribution * 0.3

        return min(stress_score, 1.0)

    def classify_emotion(self, features: AudioFeatures, stress_level: float) -> Tuple[EmotionType, float]:
        """
        Classify emotion from audio features using rule-based approach.
        Returns emotion type and confidence.
        """
        pitch = features.pitch_hz
        energy = features.energy
        speaking_rate = features.speaking_rate_wpm

        # High pitch + High energy = ANGRY
        if pitch > 350 and energy > 0.5:
            confidence = min((pitch - 350) / 150 + energy, 1.0) * 0.8
            return EmotionType.ANGRY, confidence

        # Low pitch + Low energy = SAD
        if pitch < 200 and energy < 0.3:
            confidence = min((200 - pitch) / 200 + (0.3 - energy) / 0.3, 1.0) * 0.7
            return EmotionType.SAD, confidence

        # High stress + Fast speech + High pitch = FEAR
        if stress_level > 0.6 and speaking_rate > 180 and pitch > 350:
            confidence = min(stress_level + (speaking_rate - 180) / 100, 1.0) * 0.75
            return EmotionType.FEAR, confidence

        # Very high energy + pitch variation = SURPRISE
        if energy > 0.7 and features.spectral_centroid > 3000:
            confidence = energy * 0.6
            return EmotionType.SURPRISE, confidence

        # Low energy + moderate pitch = neutral/default
        if energy < 0.4 and stress_level < 0.3:
            return EmotionType.NEUTRAL, 0.6

        # Default
        return EmotionType.NEUTRAL, 0.5

    def calculate_crisis_score(self, features: AudioFeatures, emotion: EmotionType, stress: float) -> float:
        """
        Calculate crisis score from audio.
        High stress + negative emotions = potential crisis.
        """
        crisis_score = 0.0

        # Negative emotions contribute to crisis
        negative_emotions = [EmotionType.SAD, EmotionType.ANGRY, EmotionType.FEAR, EmotionType.DISGUST]
        if emotion in negative_emotions:
            crisis_score += 0.3

        # High stress contributes
        if stress > 0.7:
            crisis_score += 0.4
        elif stress > 0.5:
            crisis_score += 0.2

        # Very high pitch (shouting/crying) indicates distress
        if features.pitch_hz > 500:
            crisis_score += 0.3
        elif features.pitch_hz > 400:
            crisis_score += 0.15

        # Very low energy (monotone) can indicate depression
        if features.energy < 0.15:
            crisis_score += 0.2

        return min(crisis_score, 1.0)

    def process(self, audio_request: AudioRequest) -> AudioInput:
        """
        Process audio input through the complete pipeline.
        Returns AudioInput with all analysis results.
        """
        start_time = time.time()

        # Load audio
        y, sr = self.load_audio(audio_request.audio_data)

        # Check if voice detected
        voice_detected = len(y) > 0 and np.max(np.abs(y)) > 0.01

        if not voice_detected:
            return AudioInput(
                pitch_hz=0.0,
                energy=0.0,
                speaking_rate_wpm=0.0,
                voice_stress_level=0.0,
                emotion=EmotionType.NEUTRAL,
                emotion_confidence=0.0,
                crisis_score=0.0,
                transcript=None,
                processing_time_ms=(time.time() - start_time) * 1000,
                voice_detected=False
            )

        # Extract features
        features = self.extract_features(y, sr)

        # Detect stress
        stress_level = self.detect_stress(features)

        # Classify emotion
        emotion, emotion_confidence = self.classify_emotion(features, stress_level)

        # Calculate crisis score
        crisis_score = self.calculate_crisis_score(features, emotion, stress_level)

        # Transcribe audio to text
        transcript = self.transcribe_audio(y)

        processing_time = (time.time() - start_time) * 1000

        return AudioInput(
            pitch_hz=features.pitch_hz,
            energy=features.energy,
            speaking_rate_wpm=features.speaking_rate_wpm,
            voice_stress_level=stress_level,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            crisis_score=crisis_score,
            transcript=transcript,
            processing_time_ms=processing_time,
            voice_detected=True
        )


# Global instance
audio_processor = AudioProcessor()
