"""
MHEART Model Download Script
Downloads required ML models.
"""

import os
import sys

def download_nltk_data():
    """Download NLTK data"""
    import nltk
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("NLTK data downloaded.")

def download_huggingface_models():
    """Download HuggingFace models"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sentence_transformers import SentenceTransformer

    print("Downloading DistilBERT sentiment model...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)
    print("DistilBERT downloaded.")

    print("Downloading emotion classifier...")
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Emotion classifier downloaded.")

    print("Downloading sentence transformer...")
    SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence transformer downloaded.")

def download_ollama_models():
    """Download Ollama models"""
    import subprocess
    print("Downloading Mistral model for Ollama...")
    subprocess.run(["ollama", "pull", "mistral"])
    print("Mistral downloaded.")

if __name__ == "__main__":
    print("MHEART Model Downloader")
    print("=" * 50)

    try:
        download_nltk_data()
        download_huggingface_models()
        download_ollama_models()

        print("=" * 50)
        print("All models downloaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)