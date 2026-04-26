"""
MHEART Video Processing Module
Handles all video-based input processing including:
- Face detection using MediaPipe
- Facial landmark extraction
- Emotion recognition using DeepFace
- Crisis detection from facial expressions
"""

import time
import io
import base64
from typing import Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

from backend.schemas.models import VideoInput, VideoRequest, EmotionType
from backend.core.config import settings


@dataclass
class FaceLandmarks:
    """Facial landmark data"""
    landmarks: List[List[float]]  # 468 points x,y,z
    left_eye: List[float]
    right_eye: List[float]
    mouth: List[float]
    left_eyebrow: List[float]
    right_eyebrow: List[float]


class VideoProcessor:
    """
    Video processing module for mental health detection.
    Uses MediaPipe for face detection/landmarks and DeepFace for emotion recognition.
    """

    def __init__(self):
        self.mp_face_mesh = None
        self.mp_drawing = None
        self.deepface_available = False
        self._initialize_models()

    def _initialize_models(self):
        """Initialize MediaPipe and DeepFace"""
        try:
            import mediapipe as mp
            # Check if solutions module exists (newer mediapipe versions)
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                print("[VIDEO] MediaPipe face mesh loaded successfully")
            else:
                # Try alternate import pattern
                from mediapipe.python.solutions import face_mesh as mp_fm
                from mediapipe.python.solutions import drawing_utils as mp_dw
                self.mp_face_mesh = mp_fm
                self.mp_drawing = mp_dw
                print("[VIDEO] MediaPipe face mesh loaded successfully (alt import)")
        except ImportError:
            print("[VIDEO] Warning: MediaPipe not available")
        except Exception as e:
            print(f"[VIDEO] Warning: MediaPipe initialization failed: {e}")
            self.mp_face_mesh = None
            self.mp_drawing = None

        try:
            from deepface import DeepFace
            self.deepface_analyze = DeepFace.analyze
            self.deepface_available = True
            print("[VIDEO] DeepFace loaded successfully")
        except ImportError:
            print("[VIDEO] Warning: DeepFace not available, using fallback")
        except Exception as e:
            print(f"[VIDEO] Warning: DeepFace not available: {e}")
            self.deepface_available = False

    def decode_image(self, frame_data: bytes) -> np.ndarray:
        """
        Decode image from bytes or base64.
        Returns RGB image array.
        """
        try:
            # Try base64 decode first
            if isinstance(frame_data, str):
                # Remove data URL prefix if present
                if ',' in frame_data:
                    frame_data = frame_data.split(',')[1]
                frame_data = base64.b64decode(frame_data)

            # Open with PIL
            image = Image.open(io.BytesIO(frame_data))

            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return np.array(image)

        except Exception as e:
            print(f"[VIDEO] Error decoding image: {e}")
            return None

    def detect_face(self, image: np.ndarray) -> Tuple[bool, Optional[FaceLandmarks]]:
        """
        Detect face and extract landmarks using MediaPipe.
        Returns face detected boolean and landmarks.
        """
        if self.mp_face_mesh is None:
            return False, None

        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:

                results = face_mesh.process(image)

                if not results.multi_face_landmarks:
                    return False, None

                face_landmarks = results.multi_face_landmarks[0]

                # Extract key landmarks
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                # Get specific facial regions (approximate indices)
                left_eye = landmarks[33]  # Left eye corner
                right_eye = landmarks[263]  # Right eye corner
                mouth = landmarks[13]  # Upper lip center
                left_eyebrow = landmarks[70]  # Left eyebrow
                right_eyebrow = landmarks[300]  # Right eyebrow

                face_data = FaceLandmarks(
                    landmarks=landmarks,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    mouth=mouth,
                    left_eyebrow=left_eyebrow,
                    right_eyebrow=right_eyebrow
                )

                return True, face_data

        except Exception as e:
            print(f"[VIDEO] Face detection error: {e}")
            return False, None

    def analyze_emotion_deepface(self, image: np.ndarray) -> Tuple[EmotionType, float, dict]:
        """
        Analyze emotion using DeepFace.
        Returns dominant emotion, confidence, and all emotion scores.
        """
        if not self.deepface_available:
            return EmotionType.NEUTRAL, 0.0, {}

        try:
            # DeepFace analyze returns a list
            result = self.deepface_analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if not result:
                return EmotionType.NEUTRAL, 0.0, {}

            emotions = result[0]['emotion']

            # Map DeepFace emotions to our emotions
            emotion_mapping = {
                'happy': EmotionType.HAPPY,
                'sad': EmotionType.SAD,
                'angry': EmotionType.ANGRY,
                'fear': EmotionType.FEAR,
                'surprise': EmotionType.SURPRISE,
                'disgust': EmotionType.DISGUST,
                'neutral': EmotionType.NEUTRAL
            }

            # Find dominant
            dominant = max(emotions.items(), key=lambda x: x[1])
            dominant_emotion = emotion_mapping.get(dominant[0], EmotionType.NEUTRAL)
            confidence = dominant[1]

            # Map all scores
            mapped_emotions = {}
            for emo, score in emotions.items():
                mapped_emo = emotion_mapping.get(emo, EmotionType.NEUTRAL)
                mapped_emotions[mapped_emo] = score / 100.0  # Normalize to 0-1

            return dominant_emotion, confidence / 100.0, mapped_emotions

        except Exception as e:
            print(f"[VIDEO] DeepFace emotion analysis error: {e}")
            return EmotionType.NEUTRAL, 0.0, {}

    def calculate_eye_contact(self, landmarks: FaceLandmarks, image_shape: tuple) -> float:
        """
        Calculate eye contact ratio from landmarks.
        Returns ratio between 0 and 1.
        """
        try:
            h, w = image_shape[:2]

            # Get eye positions in pixel coordinates
            left_eye_x = landmarks.left_eye[0] * w
            right_eye_x = landmarks.right_eye[0] * w

            # Eye contact is determined by if eyes are looking at camera
            # Simplified: check if eyes are roughly horizontal and centered
            eye_distance = abs(right_eye_x - left_eye_x)
            face_width_estimate = w * 0.4  # Approximate face width ratio

            # Normal eye contact ratio
            eye_contact = min(eye_distance / face_width_estimate, 1.0)

            return eye_contact

        except Exception as e:
            print(f"[VIDEO] Eye contact calculation error: {e}")
            return 0.5

    def calculate_facial_activity(self, landmarks: FaceLandmarks) -> float:
        """
        Calculate facial activity level from landmarks.
        Returns activity ratio between 0 and 1.
        """
        try:
            # Calculate variability in key facial regions
            # Higher activity = more facial movement (positive indicator)

            left_eye_y = landmarks.left_eye[1]
            right_eye_y = landmarks.right_eye[1]
            mouth_y = landmarks.mouth[1]

            # Eye openness (approximate)
            eye_openness = abs(left_eye_y - right_eye_y)

            # Mouth position relative to eyes
            mouth_position = mouth_y - min(left_eye_y, right_eye_y)

            # Normalize (these are rough heuristics)
            activity = (eye_openness * 2 + mouth_position) / 3

            return min(activity, 1.0)

        except Exception as e:
            print(f"[VIDEO] Facial activity calculation error: {e}")
            return 0.5

    def detect_crisis_from_face(
        self,
        emotion: EmotionType,
        emotion_confidence: float,
        facial_activity: float
    ) -> float:
        """
        Detect crisis indicators from facial expressions.
        Fear + high confidence = potential crisis.
        """
        crisis_score = 0.0

        # High fear with high confidence = crisis
        if emotion == EmotionType.FEAR and emotion_confidence > 0.7:
            crisis_score += 0.5

        # High sadness with high confidence = medium crisis
        if emotion == EmotionType.SAD and emotion_confidence > 0.7:
            crisis_score += 0.3

        # Very low facial activity (flat affect) = possible depression
        if facial_activity < 0.2:
            crisis_score += 0.2

        # Panic expression heuristics
        # Wide eyes (high facial activity in eye region)
        if facial_activity > 0.8:
            crisis_score += 0.2

        return min(crisis_score, 1.0)

    def process(self, video_request: VideoRequest) -> VideoInput:
        """
        Process video frame through the complete pipeline.
        Returns VideoInput with all analysis results.
        """
        start_time = time.time()

        # Decode image
        image = self.decode_image(video_request.frame_data)

        if image is None:
            return VideoInput(
                face_detected=False,
                landmark_points=[],
                emotion=EmotionType.NEUTRAL,
                emotion_confidence=0.0,
                all_emotions={},
                crisis_score=0.0,
                eye_contact_ratio=0.0,
                facial_activity=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Detect face and landmarks
        face_detected, landmarks = self.detect_face(image)

        if not face_detected:
            return VideoInput(
                face_detected=False,
                landmark_points=[],
                emotion=EmotionType.NEUTRAL,
                emotion_confidence=0.0,
                all_emotions={},
                crisis_score=0.0,
                eye_contact_ratio=0.0,
                facial_activity=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Analyze emotion
        emotion, emotion_confidence, all_emotions = self.analyze_emotion_deepface(image)

        # Calculate eye contact
        eye_contact_ratio = self.calculate_eye_contact(landmarks, image.shape)

        # Calculate facial activity
        facial_activity = self.calculate_facial_activity(landmarks)

        # Detect crisis
        crisis_score = self.detect_crisis_from_face(emotion, emotion_confidence, facial_activity)

        # Convert landmarks to list format
        landmark_points = [[float(p[0]), float(p[1]), float(p[2])] for p in landmarks.landmarks]

        processing_time = (time.time() - start_time) * 1000

        return VideoInput(
            face_detected=True,
            landmark_points=landmark_points,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            all_emotions=all_emotions,
            crisis_score=crisis_score,
            eye_contact_ratio=eye_contact_ratio,
            facial_activity=facial_activity,
            processing_time_ms=processing_time
        )


# Global instance
video_processor = VideoProcessor()
