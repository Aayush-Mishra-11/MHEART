"""
MHEART RAG Module (Retrieval-Augmented Generation)
Handles knowledge base creation and retrieval for context-aware responses.
"""

from typing import List, Optional
from pathlib import Path

from backend.schemas.models import RetrievedContext
from backend.core.config import settings


# Default knowledge base content
DEFAULT_GUIDELINES = """
# Mental Health Support Guidelines

## Crisis Intervention
When someone expresses thoughts of self-harm or suicide:
- Take them seriously
- Ask directly: "Are you thinking about killing yourself?"
- Do not leave them alone
- Help them connect with professional help
- Provide crisis hotline numbers (988 in US)

## Depression Support
Signs of depression: persistent sadness, loss of interest, fatigue, sleep changes
How to help:
- Listen without judgment
- Encourage professional help
- Suggest small, achievable activities
- Be patient - recovery takes time
- Remind them their feelings are valid

## Anxiety Support
Signs of anxiety: excessive worry, restlessness, physical symptoms
Grounding techniques:
- 5-4-3-2-1: Name 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste
- Deep breathing: 4 seconds in, 4 seconds hold, 4 seconds out
- Progressive muscle relaxation

## Emotional Validation
Always validate before giving advice:
- "That sounds really hard"
- "I can see why you'd feel that way"
- "Your feelings are understandable given what you're going through"
- Avoid: "It could be worse", "Just think positive", "You should..."

## Active Listening
- Reflect back what you hear
- Ask open-ended questions
- Don't interrupt or finish sentences
- Show you're paying attention
- Summarize to check understanding

## When to Escalate
Seek immediate help if:
- Threats of self-harm or suicide
- Self-harm behaviors
- Psychotic symptoms (hallucinations, delusions)
- Substance abuse combined with distress
- Any situation where safety is a concern
"""


class RAGRetriever:
    """
    Retrieval-Augmented Generation module.
    Handles knowledge base storage and retrieval.
    """

    def __init__(self):
        self.vector_db = None
        self.embeddings = None
        self._initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize the vector database"""
        try:
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.vectorstores import Chroma
            from langchain.embeddings import SentenceTransformerEmbeddings

            # Create embeddings model
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            # Load or create knowledge base
            kb_path = settings.knowledge_base_path
            if kb_path.exists():
                with open(kb_path, 'r') as f:
                    knowledge_text = f.read()
            else:
                knowledge_text = DEFAULT_GUIDELINES

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            texts = text_splitter.split_text(knowledge_text)

            # Create vector database
            self.vector_db = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=str(settings.vector_db_path)
            )

            self._initialized = True
            print(f"[RAG] Vector database initialized with {len(texts)} chunks")

        except ImportError as e:
            print(f"[RAG] Warning: LangChain not available: {e}")
            self._initialized = False
        except Exception as e:
            print(f"[RAG] Warning: Could not initialize vector DB: {e}")
            self._initialized = False

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> List[RetrievedContext]:
        """
        Retrieve relevant context from knowledge base.
        Returns list of RetrievedContext objects.
        """
        if not self._initialized or self.vector_db is None:
            return self._fallback_retrieve(query)

        try:
            docs = self.vector_db.similarity_search(query, k=k)

            results = []
            for doc in docs:
                results.append(RetrievedContext(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "knowledge_base"),
                    relevance_score=doc.metadata.get("distance", 0.0)
                ))

            return results

        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return self._fallback_retrieve(query)

    def _fallback_retrieve(self, query: str) -> List[RetrievedContext]:
        """
        Fallback retrieval when vector DB is not available.
        Simple keyword-based retrieval from default guidelines.
        """
        query_lower = query.lower()
        results = []

        # Simple keyword matching
        keywords_map = {
            "suicide": "Crisis Intervention - Always take suicidal thoughts seriously. Ask directly about safety.",
            "self-harm": "Crisis Intervention - Self-harm requires immediate professional support.",
            "sad": "Depression Support - Listen without judgment and encourage professional help.",
            "depression": "Depression Support - Recovery takes time. Be patient and supportive.",
            "anxiety": "Anxiety Support - Try grounding techniques like 5-4-3-2-1 breathing.",
            "worried": "Anxiety Support - Deep breathing and being present can help.",
            "alone": "Emotional Validation - Your feelings are valid. Consider reaching out for support.",
            "tired": "Depression Support - Fatigue is common. Small activities can help build energy.",
            "angry": "Emotional Validation - Anger is a valid emotion. Let's explore what's underneath.",
            "scared": "Anxiety Support - Fear is natural. Grounding techniques can help you feel more present."
        }

        for keyword, guidance in keywords_map.items():
            if keyword in query_lower:
                results.append(RetrievedContext(
                    content=guidance,
                    source="fallback_guidelines",
                    relevance_score=0.8
                ))

        if not results:
            results.append(RetrievedContext(
                content="Emotional Validation - Your feelings are valid and worth discussing.",
                source="general_guidelines",
                relevance_score=0.5
            ))

        return results[:3]

    def add_document(self, text: str, source: str = "user_added"):
        """Add a new document to the knowledge base"""
        if not self._initialized or self.vector_db is None:
            print("[RAG] Cannot add document - vector DB not initialized")
            return

        try:
            self.vector_db.add_texts(
                texts=[text],
                metadatas=[{"source": source}]
            )
            print(f"[RAG] Document added from {source}")
        except Exception as e:
            print(f"[RAG] Error adding document: {e}")


# Global instance
rag_retriever = RAGRetriever()
