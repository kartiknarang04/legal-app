# Legal Document Analyzer Setup Script

echo "🏗️  Setting up Legal Document Analyzer..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install

# Create backend virtual environment
echo "🐍 Setting up Python virtual environment..."
cd backend
python3 -m venv venv

# Activate virtual environment and install dependencies
echo "📦 Installing backend dependencies..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/Linux/macOS
    source venv/bin/activate
fi

pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/spacy_model
mkdir -p models/legalbert
mkdir -p data/rag_index

# Copy environment template
if [ ! -f .env ]; then
    echo "⚙️  Creating environment configuration..."
    cp .env.example .env
    echo "✏️  Please edit backend/.env with your model paths and API keys"
fi

cd ..

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your trained models in backend/models/"
echo "2. Edit backend/.env with your configuration"
echo "3. Run 'npm run dev' to start the application"
echo ""
echo "For more information, see README.md"
