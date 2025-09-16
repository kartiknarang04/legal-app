# Development startup script

echo "🚀 Starting Legal Document Analyzer in development mode..."

# Check if backend virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Backend virtual environment not found. Run 'npm run setup' first."
    exit 1
fi

# Start both frontend and backend concurrently
npm run dev
