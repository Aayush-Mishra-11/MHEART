"""
MHEART LLM Module
Handles local LLM generation using Ollama.
Integrates with RAG for context-aware responses.
"""

import time
from typing import List, Optional

from backend.schemas.models import (
    PersonaConfig, LLMResponse, RetrievedContext, PersonaType
)
from backend.core.config import settings


class LLMGenerator:
    """
    Local LLM generation using Ollama.
    Integrates persona system and RAG context.
    """

    def __init__(self):
        self.ollama = None
        self.model = settings.ollama.model
        self.temperature = settings.ollama.temperature
        self.max_tokens = settings.ollama.max_tokens
        self._initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize Ollama connection"""
        try:
            from langchain.llms import Ollama
            self.ollama = Ollama(
                model=self.model,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
            self._initialized = True
            print(f"[LLM] Ollama initialized with model: {self.model}")
        except ImportError:
            print("[LLM] Warning: LangChain Ollama not available")
            self._initialized = False
        except Exception as e:
            print(f"[LLM] Warning: Could not initialize Ollama: {e}")
            self._initialized = False

    def is_available(self) -> bool:
        """Check if LLM is available"""
        if not self._initialized or self.ollama is None:
            return False

        try:
            # Simple test call
            self.ollama("Hello")
            return True
        except:
            return False

    def build_prompt(
        self,
        user_input: str,
        persona: PersonaConfig,
        context: List[RetrievedContext]
    ) -> str:
        """
        Build prompt with persona and context.
        """
        # Format context
        context_text = ""
        if context:
            context_lines = []
            for ctx in context:
                context_lines.append(f"- {ctx.content}")
            context_text = "\n".join(context_lines)
        else:
            context_text = "General emotional support"

        # Build full prompt
        prompt = f"""System: {persona.system_prompt}

Relevant Guidelines:
{context_text}

User: {user_input}

Response (follow the persona guidelines above):"""

        return prompt

    def generate(
        self,
        user_input: str,
        persona: PersonaConfig,
        context: List[RetrievedContext] = None
    ) -> LLMResponse:
        """
        Generate response using local LLM.
        Returns LLMResponse with generated text.
        """
        start_time = time.time()

        if not self._initialized or self.ollama is None:
            return self._fallback_response(user_input, persona, context, start_time)

        try:
            prompt = self.build_prompt(user_input, persona, context or [])

            response_text = self.ollama(prompt)

            generation_time = (time.time() - start_time) * 1000

            return LLMResponse(
                response_text=response_text,
                persona_used=persona.persona_type,
                context_used=context or [],
                generation_time_ms=generation_time,
                tokens_used=len(response_text.split())  # Approximate
            )

        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return self._fallback_response(user_input, persona, context, start_time)

    def _fallback_response(
        self,
        user_input: str,
        persona: PersonaConfig,
        context: List[RetrievedContext],
        start_time: time.time
    ) -> LLMResponse:
        """
        Fallback response when LLM is not available.
        Provides basic rule-based responses.
        """
        user_input_lower = user_input.lower()

        # Simple rule-based responses based on persona and content
        if persona.persona_type == PersonaType.CLINICAL_PSYCHOLOGIST:
            if any(word in user_input_lower for word in ['suicide', 'kill', 'die', 'end it']):
                response = (
                    "I'm concerned about what you're sharing. "
                    "Can you tell me if you're having thoughts of hurting yourself? "
                    "If you're in immediate danger, please call 988 (US) or your local emergency services. "
                    "You deserve support, and I'm here to listen."
                )
            elif any(word in user_input_lower for word in ['sad', 'depressed', 'hopeless']):
                response = (
                    "I hear you, and I want you to know that what you're feeling is valid. "
                    "Depression can make everything feel overwhelming. "
                    "Have you been able to speak with a mental health professional? "
                    "There are people who want to help you through this."
                )
            else:
                response = (
                    "Thank you for sharing with me. Your safety and wellbeing are important. "
                    "Can you tell me more about what you're experiencing? "
                    "I'm here to listen without judgment."
                )

        elif persona.persona_type == PersonaType.EMPATHETIC_LISTENER:
            if any(word in user_input_lower for word in ['sad', 'hurt', 'pain']):
                response = (
                    "That sounds really difficult. I can hear how much this is weighing on you. "
                    "You don't have to carry this alone. "
                    "What's one thing that's been the hardest to deal with?"
                )
            elif any(word in user_input_lower for word in ['angry', 'frustrated']):
                response = (
                    "It sounds like you're feeling really frustrated. "
                    "Those feelings are completely valid. "
                    "Would you like to talk about what's been making you angry?"
                )
            else:
                response = (
                    "Thank you for opening up. I'm here to listen. "
                    "Take your time, and share whatever feels right. "
                    "I hear you."
                )

        else:  # FRIENDLY_COMPANION
            response = (
                "It's great to chat with you today! "
                "How are you feeling right now? "
                "I'm here if you want to talk about anything."
            )

        generation_time = (time.time() - start_time) * 1000

        return LLMResponse(
            response_text=response,
            persona_used=persona.persona_type,
            context_used=context or [],
            generation_time_ms=generation_time
        )


# Global instance
llm_generator = LLMGenerator()