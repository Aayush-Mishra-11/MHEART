#!/bin/bash
# MHEART Setup Script

echo "Setting up MHEART..."

# Create virtual environment if it doesn't exist
if [ ! -d "mheart_env" ]; then
    echo "Creating virtual environment..."
    python -m venv mheart_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source mheart_env/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Go back to root
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install

# Install Ollama (if not installed)
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    # See https://github.com/ollama/ollama for installation instructions
    echo "Please install Ollama manually from https://github.com/ollama/ollama"
else
    echo "Pulling Mistral model..."
    ollama pull mistral
fi

echo ""
echo "Setup complete!"
echo ""
echo "To start the backend:"
echo "  cd backend"
echo "  uvicorn backend.main:app --reload"
echo ""
echo "To start the frontend:"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"