#!/bin/bash

# Build and deployment script for hand-object inspection detection
# Bu script, projeyi build eder ve çeşitli deployment seçenekleri sunar

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fpi-mpi-hand-detection"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "v0.1.0")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REGISTRY="localhost:5000"  # Change to your registry

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
🔍 Hand-Object Inspection Detection - Build & Deploy Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  setup           Setup development environment
  build           Build Docker image
  test            Run tests
  deploy-local    Deploy locally with docker-compose
  deploy-prod     Deploy to production
  clean           Clean build artifacts
  help            Show this help

Options:
  --push          Push Docker image to registry
  --no-cache      Build without Docker cache
  --gpu           Enable GPU support
  --dev           Development mode
  --version       Show version information

Examples:
  $0 setup                    # Setup development environment
  $0 build --push             # Build and push Docker image
  $0 deploy-local --gpu       # Deploy locally with GPU support
  $0 test                     # Run all tests

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed!"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed!"
        exit 1
    fi
    
    log_info "All dependencies are available ✅"
}

setup_environment() {
    log_info "Setting up development environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
    
    # Create directories
    log_info "Creating project directories..."
    mkdir -p data/{raw,processed,events,logs,models}
    mkdir -p runs/{train,val,test}
    
    # Copy example configs
    if [ ! -f "configs/logic.yaml" ] && [ -f "configs/logic.yaml.example" ]; then
        log_info "Copying example configuration..."
        cp configs/logic.yaml.example configs/logic.yaml
    fi
    
    if [ ! -f "configs/model.yaml" ] && [ -f "configs/model.yaml.example" ]; then
        cp configs/model.yaml.example configs/model.yaml
    fi
    
    # Setup pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi
    
    log_info "Development environment setup complete! ✅"
    log_info "Activate with: source venv/bin/activate"
}

build_docker() {
    local push=false
    local no_cache=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --push)
                push=true
                shift
                ;;
            --no-cache)
                no_cache=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "Building Docker image..."
    
    # Build arguments
    local build_args="--build-arg VERSION=${VERSION} --build-arg BUILD_DATE=${BUILD_DATE}"
    
    if [ "$no_cache" = true ]; then
        build_args="$build_args --no-cache"
    fi
    
    # Build image
    docker build $build_args -t ${PROJECT_NAME}:${VERSION} -t ${PROJECT_NAME}:latest .
    
    # Tag for registry
    if [ "$push" = true ]; then
        log_info "Tagging for registry..."
        docker tag ${PROJECT_NAME}:${VERSION} ${REGISTRY}/${PROJECT_NAME}:${VERSION}
        docker tag ${PROJECT_NAME}:latest ${REGISTRY}/${PROJECT_NAME}:latest
        
        log_info "Pushing to registry..."
        docker push ${REGISTRY}/${PROJECT_NAME}:${VERSION}
        docker push ${REGISTRY}/${PROJECT_NAME}:latest
    fi
    
    log_info "Docker build complete! ✅"
    docker images | grep ${PROJECT_NAME}
}

run_tests() {
    log_info "Running tests..."
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Run pytest
    if command -v pytest &> /dev/null; then
        log_info "Running unit tests..."
        pytest tests/ -v --cov=src --cov-report=html --cov-report=term
        
        log_info "Running linting..."
        if command -v flake8 &> /dev/null; then
            flake8 src/ tests/
        fi
        
        if command -v mypy &> /dev/null; then
            mypy src/
        fi
        
        log_info "Tests completed! ✅"
    else
        log_warn "pytest not found, skipping tests"
    fi
}

deploy_local() {
    local gpu=false
    local dev=false
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                gpu=true
                shift
                ;;
            --dev)
                dev=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "Deploying locally with docker-compose..."
    
    # Prepare environment file
    cat > .env << EOF
PROJECT_NAME=${PROJECT_NAME}
VERSION=${VERSION}
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
EOF
    
    # Select compose file
    local compose_file="docker-compose.yml"
    if [ "$gpu" = true ]; then
        compose_file="$compose_file:docker-compose.gpu.yml"
    fi
    
    if [ "$dev" = true ]; then
        compose_file="$compose_file:docker-compose.dev.yml"
    fi
    
    # Start services
    log_info "Starting services..."
    docker-compose -f $compose_file up -d
    
    # Wait for services
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_info "API service is healthy! ✅"
        log_info "API documentation: http://localhost:8000/docs"
    else
        log_warn "API service health check failed"
    fi
    
    log_info "Local deployment complete! ✅"
}

deploy_production() {
    log_info "Deploying to production..."
    
    # Ensure we have the latest image
    docker pull ${REGISTRY}/${PROJECT_NAME}:${VERSION}
    
    # Deploy with production compose
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    
    log_info "Production deployment complete! ✅"
}

clean_artifacts() {
    log_info "Cleaning build artifacts..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove Docker images
    docker rmi ${PROJECT_NAME}:${VERSION} 2>/dev/null || true
    docker rmi ${PROJECT_NAME}:latest 2>/dev/null || true
    
    # Clean Python cache
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -delete
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    
    # Clean build directories
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    
    log_info "Cleanup complete! ✅"
}

show_version() {
    cat << EOF
🔍 Hand-Object Inspection Detection
Version: ${VERSION}
Build Date: ${BUILD_DATE}
Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
EOF
}

# Main script logic
main() {
    case "${1:-help}" in
        setup)
            check_dependencies
            setup_environment
            ;;
        build)
            shift
            check_dependencies
            build_docker "$@"
            ;;
        test)
            run_tests
            ;;
        deploy-local)
            shift
            check_dependencies
            deploy_local "$@"
            ;;
        deploy-prod)
            check_dependencies
            deploy_production
            ;;
        clean)
            clean_artifacts
            ;;
        version)
            show_version
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
