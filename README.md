# MHEART - Mental Health Emotion Analysis & Response Terminal

## Overview

MHEART is a multimodal mental health crisis detection system designed to analyze and respond to emotional distress through advanced AI-powered analysis. The project leverages cutting-edge machine learning models to provide real-time mental health support and crisis intervention.

## End Goal

To create an accessible, intelligent mental health companion that can:
- Detect emotional distress and crisis situations through multiple input modalities
- Provide immediate, empathetic responses and appropriate resources
- Bridge the gap between individuals in need and professional mental health support
- Reduce barriers to mental health assistance through technology

## Objectives

- **Multimodal Analysis**: Process text, voice, and potentially visual inputs to understand emotional state
- **Crisis Detection**: Identify signs of mental health crises, self-harm ideation, and emotional distress
- **Empathetic Response**: Generate contextually appropriate, supportive responses using LLMs
- **Resource Connection**: Connect users with appropriate mental health resources and professional help when needed
- **Privacy-First**: Ensure user data is handled with the utmost confidentiality and security

## Why MHEART?

Mental health challenges affect millions worldwide, yet barriers to access remain significant:
- **Stigma** prevents many from seeking help
- **Cost** of therapy and counseling is prohibitive
- **Availability** of mental health professionals is limited
- **Wait times** for appointments can be weeks or months

MHEART aims to provide an immediate, judgment-free point of contact for individuals struggling with their mental health, offering support and guidance 24/7.

## Project Structure

```
MHEART/
├── backend/          # FastAPI backend with AI/ML processing
├── frontend/         # React-based user interface
├── scripts/          # Utility and setup scripts
├── tests/            # Test suite
├── data/             # Data storage and vector databases
├── docker-compose.yml
├── Dockerfile.backend
└── Dockerfile.frontend
```

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, Hugging Face Transformers
- **Frontend**: React, Modern UI/UX
- **AI/ML**: Emotion detection models, LLM integration (Claude API)
- **Deployment**: Docker, Docker Compose

## Current Status

**This project is still in active development.** Many features are planned but not yet implemented. This is a work in progress with the core architecture being built out.

### What's Working
- Basic backend API structure
- Foundation for multimodal analysis
- Initial frontend setup

### Coming Soon
- Full multimodal input processing
- Enhanced crisis detection algorithms
- Complete UI/UX implementation
- Integration with crisis helpline APIs

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js (for frontend)
- Docker (optional, for containerized deployment)

### Run Backend

```bash
# Create virtual environment
python -m venv mheart_env
source mheart_env/bin/activate  # On Windows: mheart_env\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8003
```

### Run Frontend

```bash
cd frontend
npm install
npm start
```

### Docker Deployment

```bash
docker-compose up --build
```

## API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/multimodal` - Main analysis endpoint for processing inputs

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=backend
```

## Contributing

This is an open mental health initiative. Contributions, suggestions, and feedback are welcome.

## Disclaimer

**MHEART is not a replacement for professional mental health care.** If you are experiencing a mental health crisis, please contact:
- **National Suicide Prevention Lifeline**: 988 (US)
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

## License

This project is open source. See LICENSE file for details.

---

*Built with the goal of making mental health support more accessible to everyone.*
