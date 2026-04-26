"""
MHEART API Routes
FastAPI routes for all endpoints.
"""

import base64
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.schemas.models import (
    TextRequest, ChatRequest, ChatResponse, EmotionResponse,
    HealthResponse, PipelineResult, EmotionType, CrisisType, DiscrepancyType,
    AudioRequest, VideoRequest
)
from backend.services.pipeline import mheart_pipeline
from backend.modules.llm.processor import llm_generator


# Router instances
router = APIRouter()


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and model status"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        ollama_connected=llm_generator.is_available()
    )


# ============================================================================
# TEXT CHAT
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process text chat message.
    Returns response with emotion detection and crisis alert.
    """
    try:
        start_time = time.time()

        # Create text request
        text_request = TextRequest(
            text=request.message,
            session_id=request.session_id
        )

        # Process through pipeline
        result = mheart_pipeline.process_text_only(text_request)

        processing_time = (time.time() - start_time) * 1000

        return ChatResponse(
            response=result.llm_response.response_text,
            emotion_detected=result.fused_emotion.dominant_emotion,
            crisis_detected=result.crisis_alert.is_crisis,
            crisis_type=result.crisis_alert.crisis_type if result.crisis_alert.is_crisis else None,
            persona_used=result.persona.persona_type,
            processing_time_ms=processing_time
        )

    except Exception as e:
        print(f"[API] Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

@router.post("/audio")
async def process_audio(
    file: UploadFile = File(...),
    session_id: Optional[UUID] = None
):
    """
    Process audio input.
    Accepts WAV/MP3 audio file.
    """
    try:
        start_time = time.time()

        # Read audio data
        audio_data = await file.read()

        # Create audio request
        audio_request = AudioRequest(
            audio_data=audio_data,
            session_id=session_id,
            format=file.content_type or "audio/wav"
        )

        # Process through pipeline
        result = mheart_pipeline.process_audio_only(audio_request)

        processing_time = (time.time() - start_time) * 1000

        return JSONResponse({
            "emotion": result.audio_result.emotion.value if result.audio_result else "unknown",
            "emotion_confidence": result.audio_result.emotion_confidence if result.audio_result else 0.0,
            "stress_level": result.audio_result.voice_stress_level if result.audio_result else 0.0,
            "crisis_detected": result.crisis_alert.is_crisis,
            "crisis_score": result.audio_result.crisis_score if result.audio_result else 0.0,
            "processing_time_ms": processing_time,
            "transcript": result.audio_result.transcript if result.audio_result else None
        })

    except Exception as e:
        print(f"[API] Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

@router.post("/video")
async def process_video(
    frame: UploadFile = File(...),
    session_id: Optional[UUID] = None
):
    """
    Process video frame.
    Accepts image file (JPEG/PNG).
    """
    try:
        start_time = time.time()

        # Read frame data
        frame_data = await frame.read()

        # Create video request
        video_request = VideoRequest(
            frame_data=frame_data,
            session_id=session_id
        )

        # Process through pipeline
        result = mheart_pipeline.process_video_only(video_request)

        processing_time = (time.time() - start_time) * 1000

        return JSONResponse({
            "face_detected": result.video_result.face_detected if result.video_result else False,
            "emotion": result.video_result.emotion.value if result.video_result else "unknown",
            "emotion_confidence": result.video_result.emotion_confidence if result.video_result else 0.0,
            "crisis_score": result.video_result.crisis_score if result.video_result else 0.0,
            "processing_time_ms": processing_time
        })

    except Exception as e:
        print(f"[API] Video processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EMOTION STATE
# ============================================================================

@router.get("/emotion", response_model=EmotionResponse)
async def get_emotion_state():
    """
    Get current emotion state.
    Note: This would typically return cached/fused state from a session.
    """
    # For now, return a placeholder
    # In production, this would track session state
    return EmotionResponse(
        text_emotion=None,
        audio_emotion=None,
        video_emotion=None,
        fused_emotion=EmotionType.NEUTRAL,
        confidence=0.0,
        is_discrepant=False,
        discrepancy_type=DiscrepancyType.NONE
    )


# ============================================================================
# CRISIS HOTLINES
# ============================================================================

@router.get("/hotlines")
async def get_hotlines():
    """Get crisis hotline numbers"""
    from backend.core.config import settings
    return {"hotlines": settings.crisis_hotlines}


# ============================================================================
# MULTIMODAL PROCESSING
# ============================================================================

class MultimodalRequest(BaseModel):
    """Request model for multimodal processing"""
    text: Optional[str] = None
    audio_data: Optional[str] = None  # Base64 encoded audio
    video_data: Optional[str] = None  # Base64 encoded image
    session_id: Optional[str] = None


class MultimodalResponse(BaseModel):
    """Response model for multimodal processing"""
    fused_emotion: str
    emotion_confidence: float
    discrepancy_detected: bool
    discrepancy_type: Optional[str] = None
    crisis_detected: bool
    crisis_type: Optional[str] = None
    crisis_score: float
    persona: str
    response_text: str
    processing_time_ms: float


@router.post("/multimodal", response_model=MultimodalResponse)
async def process_multimodal(request: MultimodalRequest):
    """
    Process text, audio, and video inputs together.
    Performs cross-modal fusion and returns unified response.
    """
    try:
        start_time = time.time()

        # Decode base64 audio if provided
        audio_req = None
        if request.audio_data:
            audio_bytes = base64.b64decode(request.audio_data)
            audio_req = AudioRequest(
                audio_data=audio_bytes,
                session_id=request.session_id,
                format="audio/wav"
            )

        # Decode base64 video if provided
        video_req = None
        if request.video_data:
            video_bytes = base64.b64decode(request.video_data)
            video_req = VideoRequest(
                frame_data=video_bytes,
                session_id=request.session_id
            )

        # Create text request if text provided
        text_req = None
        if request.text:
            text_req = TextRequest(
                text=request.text,
                session_id=request.session_id
            )

        # Process through multimodal pipeline
        result = mheart_pipeline.process_multimodal(
            text_request=text_req,
            audio_request=audio_req,
            video_request=video_req
        )

        processing_time = (time.time() - start_time) * 1000

        return MultimodalResponse(
            fused_emotion=result.fused_emotion.dominant_emotion.value,
            emotion_confidence=result.fused_emotion.confidence,
            discrepancy_detected=result.fused_emotion.is_discrepant,
            discrepancy_type=result.fused_emotion.discrepancy_type.value if result.fused_emotion.is_discrepant else None,
            crisis_detected=result.crisis_alert.is_crisis,
            crisis_type=result.crisis_alert.crisis_type.value if result.crisis_alert.is_crisis else None,
            crisis_score=result.crisis_alert.weighted_score,
            persona=result.persona.persona_type.value,
            response_text=result.llm_response.response_text,
            processing_time_ms=processing_time
        )

    except Exception as e:
        print(f"[API] Multimodal processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET CHAT
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_response(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.
    Supports text, audio (base64), and video (base64) inputs.
    """
    await manager.connect(websocket, session_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            message_type = data.get("type", "text")
            start_time = time.time()

            if message_type == "text":
                text_request = TextRequest(
                    text=data.get("content", ""),
                    session_id=session_id
                )
                result = mheart_pipeline.process_text_only(text_request)

                response = {
                    "type": "text_response",
                    "content": result.llm_response.response_text,
                    "emotion": result.fused_emotion.dominant_emotion.value,
                    "crisis_detected": result.crisis_alert.is_crisis,
                    "crisis_type": result.crisis_alert.crisis_type.value if result.crisis_alert.is_crisis else None,
                    "persona": result.persona.persona_type.value,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }

            elif message_type == "audio":
                # Decode base64 audio
                audio_b64 = data.get("content", "")
                audio_data = base64.b64decode(audio_b64)

                audio_request = AudioRequest(
                    audio_data=audio_data,
                    session_id=session_id
                )
                result = mheart_pipeline.process_audio_only(audio_request)

                response = {
                    "type": "audio_response",
                    "emotion": result.audio_result.emotion.value if result.audio_result else "unknown",
                    "stress_level": result.audio_result.voice_stress_level if result.audio_result else 0.0,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }

            elif message_type == "video":
                # Decode base64 image
                video_b64 = data.get("content", "")
                video_data = base64.b64decode(video_b64)

                video_request = VideoRequest(
                    frame_data=video_data,
                    session_id=session_id
                )
                result = mheart_pipeline.process_video_only(video_request)

                response = {
                    "type": "video_response",
                    "face_detected": result.video_result.face_detected if result.video_result else False,
                    "emotion": result.video_result.emotion.value if result.video_result else "unknown",
                    "emotion_confidence": result.video_result.emotion_confidence if result.video_result else 0.0,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }

            elif message_type == "multimodal":
                # Process all available modalities together for true CMDD
                text_req = None
                audio_req = None
                video_req = None

                # Text input
                if data.get("text"):
                    text_req = TextRequest(
                        text=data.get("text", ""),
                        session_id=session_id
                    )

                # Audio input (base64)
                if data.get("audio"):
                    audio_b64 = data.get("audio", "")
                    audio_data = base64.b64decode(audio_b64)
                    audio_req = AudioRequest(
                        audio_data=audio_data,
                        session_id=session_id
                    )

                # Video input (base64)
                if data.get("video"):
                    video_b64 = data.get("video", "")
                    video_data = base64.b64decode(video_b64)
                    video_req = VideoRequest(
                        frame_data=video_data,
                        session_id=session_id
                    )

                # Process with full multimodal pipeline
                result = mheart_pipeline.process_multimodal(
                    text_request=text_req,
                    audio_request=audio_req,
                    video_request=video_req
                )

                response = {
                    "type": "multimodal_response",
                    "content": result.llm_response.response_text,
                    "fused_emotion": result.fused_emotion.dominant_emotion.value,
                    "emotion_confidence": result.fused_emotion.confidence,
                    "is_discrepant": result.fused_emotion.is_discrepant,
                    "discrepancy_type": result.fused_emotion.discrepancy_type.value if result.fused_emotion.is_discrepant else None,
                    "crisis_detected": result.crisis_alert.is_crisis,
                    "crisis_type": result.crisis_alert.crisis_type.value if result.crisis_alert.is_crisis else None,
                    "crisis_score": result.crisis_alert.weighted_score,
                    "triggered_modalities": [m.value for m in result.crisis_alert.triggered_modalities],
                    "escalation_required": result.crisis_alert.escalation_required,
                    "persona": result.persona.persona_type.value,
                    "emotion_agreement": {k.value: v.value for k, v in result.fused_emotion.emotion_agreement.items()},
                    "processing_time_ms": (time.time() - start_time) * 1000
                }

            else:
                response = {"type": "error", "message": "Unknown message type"}

            # Send response
            await manager.send_response(session_id, response)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"[WS] Error: {e}")
        await manager.send_response(session_id, {"type": "error", "message": str(e)})
        manager.disconnect(session_id)
