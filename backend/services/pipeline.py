"""
MHEART Pipeline Module
Main processing pipeline that orchestrates all modules.
"""

import time
from typing import Optional

from backend.schemas.models import (
    TextRequest, AudioRequest, VideoRequest, PipelineResult,
    TextInput, AudioInput, VideoInput, FusedEmotion, CrisisAlert,
    PersonaConfig, LLMResponse, EmotionType
)
from backend.modules.text.processor import text_processor
from backend.modules.audio.processor import audio_processor
from backend.modules.video.processor import video_processor
from backend.modules.fusion.processor import multimodal_fusion
from backend.modules.crisis.processor import crisis_detector
from backend.modules.persona.processor import persona_modulator
from backend.modules.rag.processor import rag_retriever
from backend.modules.llm.processor import llm_generator


class MHEARTPipeline:
    """
    Main processing pipeline for MHEART.
    Orchestrates text, audio, video processing through:
    1. Individual modality processing
    2. Multimodal fusion (CMDD)
    3. Crisis detection (WCS)
    4. Persona selection (ATP)
    5. RAG retrieval
    6. LLM generation
    """

    def __init__(self):
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.video_processor = video_processor
        self.fusion = multimodal_fusion
        self.crisis = crisis_detector
        self.persona = persona_modulator
        self.rag = rag_retriever
        self.llm = llm_generator

    def process_text_only(self, text_request: TextRequest) -> PipelineResult:
        """Process text input only"""
        start_time = time.time()

        # Step 1: Text Processing
        text_result = self.text_processor.process(text_request)

        # Steps 2-6 (with text only)
        fused_emotion = self.fusion.fuse(text_input=text_result)
        crisis_alert = self.crisis.detect(text_input=text_result)
        persona = self.persona.select_persona(fused_emotion, crisis_alert)

        # RAG retrieval
        query = text_request.text
        context = self.rag.retrieve(query)

        # LLM generation
        llm_response = self.llm.generate(
            user_input=text_request.text,
            persona=persona,
            context=context
        )

        total_time = (time.time() - start_time) * 1000

        return PipelineResult(
            text_result=text_result,
            audio_result=None,
            video_result=None,
            fused_emotion=fused_emotion,
            crisis_alert=crisis_alert,
            persona=persona,
            llm_response=llm_response,
            total_processing_time_ms=total_time
        )

    def process_audio_only(self, audio_request: AudioRequest) -> PipelineResult:
        """Process audio input only"""
        start_time = time.time()

        # Audio Processing
        audio_result = self.audio_processor.process(audio_request)

        # Fusion
        fused_emotion = self.fusion.fuse(audio_input=audio_result)
        crisis_alert = self.crisis.detect(audio_input=audio_result)
        persona = self.persona.select_persona(fused_emotion, crisis_alert)

        # RAG retrieval
        transcript = audio_result.transcript or "I need someone to talk to"
        context = self.rag.retrieve(transcript)

        # LLM generation
        llm_response = self.llm.generate(
            user_input=transcript,
            persona=persona,
            context=context
        )

        total_time = (time.time() - start_time) * 1000

        return PipelineResult(
            text_result=None,
            audio_result=audio_result,
            video_result=None,
            fused_emotion=fused_emotion,
            crisis_alert=crisis_alert,
            persona=persona,
            llm_response=llm_response,
            total_processing_time_ms=total_time
        )

    def process_video_only(self, video_request: VideoRequest) -> PipelineResult:
        """Process video input only"""
        start_time = time.time()

        # Video Processing
        video_result = self.video_processor.process(video_request)

        # Fusion
        fused_emotion = self.fusion.fuse(video_input=video_result)
        crisis_alert = self.crisis.detect(video_input=video_result)
        persona = self.persona.select_persona(fused_emotion, crisis_alert)

        # RAG retrieval
        query = f"Showing emotion: {video_result.emotion.value}"
        context = self.rag.retrieve(query)

        # LLM generation
        llm_response = self.llm.generate(
            user_input=query,
            persona=persona,
            context=context
        )

        total_time = (time.time() - start_time) * 1000

        return PipelineResult(
            text_result=None,
            audio_result=None,
            video_result=video_result,
            fused_emotion=fused_emotion,
            crisis_alert=crisis_alert,
            persona=persona,
            llm_response=llm_response,
            total_processing_time_ms=total_time
        )

    def process_multimodal(
        self,
        text_request: Optional[TextRequest] = None,
        audio_request: Optional[AudioRequest] = None,
        video_request: Optional[VideoRequest] = None
    ) -> PipelineResult:
        """
        Process all available modalities.
        This is the main entry point for full multimodal processing.
        """
        start_time = time.time()

        # Step 1: Process each modality
        text_result = None
        audio_result = None
        video_result = None

        if text_request:
            text_result = self.text_processor.process(text_request)

        if audio_request:
            audio_result = self.audio_processor.process(audio_request)

        if video_request:
            video_result = self.video_processor.process(video_request)

        # Step 2: Multimodal Fusion (CMDD)
        fused_emotion = self.fusion.fuse(text_result, audio_result, video_result)

        # Step 3: Crisis Detection (WCS)
        crisis_alert = self.crisis.detect(text_result, audio_result, video_result)

        # Step 4: Persona Selection (ATP)
        persona = self.persona.select_persona(fused_emotion, crisis_alert)

        # Step 5: RAG Retrieval
        query = self._build_query(text_result, audio_result, video_result)
        context = self.rag.retrieve(query)

        # Step 6: LLM Generation
        user_input = self._build_user_input(text_result, audio_result)
        llm_response = self.llm.generate(user_input, persona, context)

        total_time = (time.time() - start_time) * 1000

        return PipelineResult(
            text_result=text_result,
            audio_result=audio_result,
            video_result=video_result,
            fused_emotion=fused_emotion,
            crisis_alert=crisis_alert,
            persona=persona,
            llm_response=llm_response,
            total_processing_time_ms=total_time
        )

    def _build_query(
        self,
        text_result: TextInput = None,
        audio_result: AudioInput = None,
        video_result: VideoInput = None
    ) -> str:
        """Build query for RAG retrieval"""
        query_parts = []

        if text_result:
            query_parts.append(text_result.raw_text)

        if audio_result and audio_result.transcript:
            query_parts.append(audio_result.transcript)

        if video_result:
            query_parts.append(f"Video emotion: {video_result.emotion.value}")

        return " ".join(query_parts) if query_parts else "emotional support"

    def _build_user_input(
        self,
        text_result: TextInput = None,
        audio_result: AudioInput = None
    ) -> str:
        """Build user input for LLM"""
        if text_result:
            return text_result.raw_text

        if audio_result and audio_result.transcript:
            return audio_result.transcript

        return "I'm not sure what to say"


# Global pipeline instance
mheart_pipeline = MHEARTPipeline()