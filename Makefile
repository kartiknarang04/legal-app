# Legal Document Analyzer Makefile

.PHONY: help setup dev build clean docker-dev docker-prod

help: ## Show this help message
	@echo "Legal Document Analyzer - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and set up the project
	@echo "🏗️  Setting up Legal Document Analyzer..."
	npm install
	cd backend && python -m venv venv
	@echo "✅ Setup complete! Edit backend/.env with your configuration."

dev: ## Start development servers
	@echo "🚀 Starting development servers..."
	npm run dev

build: ## Build the application
	@echo "🔨 Building application..."
	npm run build

clean: ## Clean build artifacts and dependencies
	@echo "🧹 Cleaning up..."
	rm -rf node_modules .next backend/venv backend/__pycache__ backend/**/__pycache__

docker-dev: ## Start development environment with Docker
	@echo "🐳 Starting Docker development environment..."
	docker-compose -f docker-compose.dev.yml up --build

docker-prod: ## Start production environment with Docker
	@echo "🐳 Starting Docker production environment..."
	docker-compose -f docker-compose.prod.yml up --build -d

docker-stop: ## Stop Docker containers
	@echo "🛑 Stopping Docker containers..."
	docker-compose down
	docker-compose -f docker-compose.dev.yml down
	docker-compose -f docker-compose.prod.yml down

docker-clean: ## Clean Docker images and containers
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

logs: ## View application logs
	@echo "📋 Viewing logs..."
	docker-compose logs -f

test-backend: ## Test backend API
	@echo "🧪 Testing backend API..."
	cd backend && python -m pytest tests/ -v

lint: ## Lint code
	@echo "🔍 Linting code..."
	npm run lint
	cd backend && flake8 . --max-line-length=100

format: ## Format code
	@echo "✨ Formatting code..."
	npm run format
	cd backend && black . --line-length=100

install-models: ## Instructions for installing models
	@echo "📦 Model Installation Instructions:"
	@echo "1. Place your trained spaCy NER model in: backend/models/spacy_model/"
	@echo "2. Place your LegalBERT model in: backend/models/legalbert/"
	@echo "3. Update backend/.env with the correct paths"
	@echo "4. Ensure models are compatible with the expected format"

check-health: ## Check application health
	@echo "🏥 Checking application health..."
	@curl -f http://localhost:8000/health || echo "Backend not responding"
	@curl -f http://localhost:3000 || echo "Frontend not responding"
