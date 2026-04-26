"""
MHEART Persona Modulator (ATP - Adaptive Therapeutic Persona)
THIS IS THE THIRD NOVEL CONTRIBUTION

Adapts the therapeutic persona based on:
1. Crisis detection (always clinical psychologist)
2. Discrepancy detection (empathetic listener)
3. Emotion intensity (decision tree)

Personas:
- clinical_psychologist: For crisis situations
- empathetic_listener: For discrepancy or high emotion
- friendly_companion: For neutral/low intensity states
"""

from backend.schemas.models import (
    FusedEmotion, CrisisAlert, PersonaConfig, PersonaType, EmotionType
)


PERSONAS = {
    PersonaType.CLINICAL_PSYCHOLOGIST: PersonaConfig(
        persona_type=PersonaType.CLINICAL_PSYCHOLOGIST,
        system_prompt="""You are a clinical psychologist. The user is in emotional distress.
Speak calmly, clearly, and professionally. Use short, direct sentences.
Prioritize safety. Ask direct questions about harm. Provide clear guidance.
Do not use casual language.

Key principles:
- Always prioritize the user's immediate safety
- Ask directly about suicidal ideation if suspected
- Provide concrete resources and next steps
- Be direct but compassionate
- Do not minimize their feelings
- Do not rush to give advice without understanding first""",
        tone="clinical",
        response_length="medium",
        max_response_length=400
    ),

    PersonaType.EMPATHETIC_LISTENER: PersonaConfig(
        persona_type=PersonaType.EMPATHETIC_LISTENER,
        system_prompt="""You are an empathetic listener. The user is sharing their feelings.
Validate their emotions. Show genuine concern. Ask thoughtful questions.
Do not rush to give advice. Be present and supportive.

Key principles:
- Validate their emotions without judgment
- Reflect back what you hear
- Ask open-ended questions
- Be fully present in the conversation
- Show you understand without trying to fix everything
- Help them explore their feelings
- Normalize their experience""",
        tone="warm",
        response_length="medium",
        max_response_length=500
    ),

    PersonaType.FRIENDLY_COMPANION: PersonaConfig(
        persona_type=PersonaType.FRIENDLY_COMPANION,
        system_prompt="""You are a friendly, supportive companion. The user is in a neutral state.
Engage warmly. Keep the conversation light but meaningful.
Be a good listener. Offer encouragement.

Key principles:
- Be warm and approachable
- Show genuine interest in their day/thoughts
- Offer positive encouragement
- Keep a friendly, conversational tone
- Suggest activities if appropriate
- Be supportive without being preachy
- Lighten the mood when appropriate""",
        tone="friendly",
        response_length="short",
        max_response_length=300
    )
}


class PersonaModulator:
    """
    Adaptive Therapeutic Persona (ATP) module.
    Selects and configures the appropriate persona based on context.
    """

    def select_persona(
        self,
        fused_emotion: FusedEmotion,
        crisis_alert: CrisisAlert
    ) -> PersonaConfig:
        """
        Select the appropriate persona based on conditions.
        Priority: Crisis > Discrepancy > Intensity
        """
        # Crisis always = Clinical Psychologist
        if crisis_alert.is_crisis:
            return PERSONAS[PersonaType.CLINICAL_PSYCHOLOGIST]

        # Discrepancy detected = Empathetic Listener
        if fused_emotion.is_discrepant:
            return PERSONAS[PersonaType.EMPATHETIC_LISTENER]

        # Based on emotion intensity
        intensity = fused_emotion.emotion_intensity

        if intensity > 0.7:
            return PERSONAS[PersonaType.CLINICAL_PSYCHOLOGIST]
        elif intensity > 0.4:
            return PERSONAS[PersonaType.EMPATHETIC_LISTENER]
        else:
            return PERSONAS[PersonaType.FRIENDLY_COMPANION]

    def modulate_response_length(
        self,
        persona: PersonaConfig,
        crisis_alert: CrisisAlert
    ) -> int:
        """
        Adjust response length based on context.
        Crisis situations may need longer responses for safety planning.
        """
        base_length = persona.max_response_length

        # Crisis may need more space for safety planning
        if crisis_alert.is_crisis:
            base_length = int(base_length * 1.3)

        return min(base_length, 800)  # Cap at 800 tokens


# Global instance
persona_modulator = PersonaModulator()
