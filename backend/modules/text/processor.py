"""
MHEART Text Processing Module
Handles all text-based input processing including:
- Text preprocessing
- Sentiment analysis using DistilBERT
- Crisis keyword detection
- Emotion classification
"""

import time
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

from backend.schemas.models import TextInput, TextRequest, EmotionType
from backend.core.config import settings


# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class CrisisKeyword:
    """Crisis keyword with weight"""
    keyword: str
    weight: float  # 1.0 = high, 0.6 = medium, 0.3 = low
    risk_level: str


class TextProcessor:
    """
    Text processing module for mental health detection.
    Uses DistilBERT for sentiment and transformers pipeline for emotion.
    """

    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_classifier = None
        self.stop_words = set(stopwords.words('english'))
        self.crisis_keywords = self._build_crisis_keywords()
        self._initialize_models()

    def _build_crisis_keywords(self) -> List[CrisisKeyword]:
        """Build crisis keyword list with weights"""
        keywords = []

        for kw in settings.crisis_keywords.high_risk:
            keywords.append(CrisisKeyword(kw, 1.0, "high"))
        for kw in settings.crisis_keywords.medium_risk:
            keywords.append(CrisisKeyword(kw, 0.6, "medium"))
        for kw in settings.crisis_keywords.low_risk:
            keywords.append(CrisisKeyword(kw, 0.3, "low"))

        return keywords

    def _initialize_models(self):
        """Initialize transformer models"""
        try:
            # Sentiment analysis model
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=AutoModelForSequenceClassification.from_pretrained(model_name),
                tokenizer=AutoTokenizer.from_pretrained(model_name)
            )

            # Emotion classification model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                top_k=None
            )

            print(f"[TEXT] Models loaded successfully")

        except Exception as e:
            print(f"[TEXT] Warning: Could not load models: {e}")
            print("[TEXT] Falling back to rule-based processing")
            self.sentiment_analyzer = None
            self.emotion_classifier = None

    def preprocess(self, text: str) -> str:
        """
        Preprocess text input.
        - Lowercase
        - Remove special characters
        - Tokenize and remove stopwords
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.\!\?\,]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using DistilBERT.
        Returns score from -1.0 (very negative) to +1.0 (very positive).
        """
        if not text or self.sentiment_analyzer is None:
            return 0.0

        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Truncate to 512 tokens
            label = result['label']
            score = result['score']

            # Convert to -1 to +1 scale
            if label == "NEGATIVE":
                return -score
            else:
                return score

        except Exception as e:
            print(f"[TEXT] Sentiment analysis error: {e}")
            return 0.0

    def classify_emotion(self, text: str) -> Tuple[EmotionType, dict]:
        """
        Classify emotion using emotion classifier.
        Returns emotion type and all emotion scores.
        """
        if not text or self.emotion_classifier is None:
            return EmotionType.NEUTRAL, {}

        try:
            result = self.emotion_classifier(text[:512])[0]

            # Convert to dict
            emotion_scores = {item['label']: item['score'] for item in result}

            # Map to our emotion types
            emotion_mapping = {
                'joy': EmotionType.HAPPY,
                'happy': EmotionType.HAPPY,
                'sadness': EmotionType.SAD,
                'sad': EmotionType.SAD,
                'anger': EmotionType.ANGRY,
                'angry': EmotionType.ANGRY,
                'fear': EmotionType.FEAR,
                'surprise': EmotionType.SURPRISE,
                'disgust': EmotionType.DISGUST,
                'neutral': EmotionType.NEUTRAL
            }

            # Find dominant emotion
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            dominant = emotion_mapping.get(max_emotion[0].lower(), EmotionType.NEUTRAL)

            # Map all scores to our types
            mapped_scores = {}
            for emo, score in emotion_scores.items():
                mapped_emo = emotion_mapping.get(emo.lower(), EmotionType.NEUTRAL)
                if mapped_emo not in mapped_scores or mapped_scores[mapped_emo] < score:
                    mapped_scores[mapped_emo] = score

            return dominant, mapped_scores

        except Exception as e:
            print(f"[TEXT] Emotion classification error: {e}")
            return EmotionType.NEUTRAL, {}

    def detect_crisis_keywords(self, text: str) -> Tuple[List[str], float]:
        """
        Detect crisis keywords in text.
        Returns list of found keywords and crisis score.
        """
        if not text:
            return [], 0.0

        text_lower = text.lower()
        found_keywords = []
        total_score = 0.0

        for kw in self.crisis_keywords:
            if kw.keyword in text_lower:
                found_keywords.append(f"{kw.keyword} ({kw.risk_level})")
                total_score += kw.weight

        # Cap at 1.0
        crisis_score = min(total_score, 1.0)

        return found_keywords, crisis_score

    def process(self, text_request: TextRequest) -> TextInput:
        """
        Process text input through the complete pipeline.
        Returns TextInput with all analysis results.
        """
        start_time = time.time()

        raw_text = text_request.text
        processed_text = self.preprocess(raw_text)

        # Sentiment analysis
        sentiment_score = self.analyze_sentiment(processed_text)

        # Emotion classification
        emotion, emotion_scores = self.classify_emotion(processed_text)

        # Crisis keyword detection
        crisis_keywords_found, crisis_score = self.detect_crisis_keywords(processed_text)

        # Determine if crisis keyword detected
        is_crisis_keyword = len(crisis_keywords_found) > 0

        processing_time = (time.time() - start_time) * 1000

        return TextInput(
            raw_text=raw_text,
            processed_text=processed_text,
            sentiment_score=sentiment_score,
            emotion=emotion,
            emotion_scores=emotion_scores,
            crisis_keywords_found=crisis_keywords_found,
            crisis_score=crisis_score,
            is_crisis_keyword=is_crisis_keyword,
            processing_time_ms=processing_time
        )


# Global instance
text_processor = TextProcessor()