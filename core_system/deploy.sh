#!/bin/bash

# Intelligent Document Processing System - Deployment Script
# Adobe India Hackathon 2025

set -e  # Exit on any error

echo "🚀 Deploying Intelligent Document Processing System"
echo "=================================================="

# Configuration
IMAGE_NAME="intelligent-document-processor"
CONTAINER_NAME="doc-processor"
VERSION="1.0.0"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check platform
    PLATFORM=$(uname -m)
    if [[ "$PLATFORM" != "x86_64" ]]; then
        log_warning "Running on $PLATFORM. Building for linux/amd64 platform."
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Check if Dockerfile exists
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    # Build the image
    docker build \\
        --platform linux/amd64 \\
        --tag "${IMAGE_NAME}:${VERSION}" \\
        --tag "${IMAGE_NAME}:latest" \\
        . || {
        log_error "Docker build failed"
        exit 1
    }
    
    log_success "Docker image built successfully"
}

# Test the image
test_image() {
    log_info "Testing Docker image..."
    
    # Create test input directory
    mkdir -p test_input test_output
    
    # Create a dummy PDF file for testing (in real scenario, use actual PDF)
    echo "Dummy PDF content for testing" > test_input/test.pdf
    
    # Run the container for testing
    docker run --rm \\
        -v "$(pwd)/test_input:/app/input:ro" \\
        -v "$(pwd)/test_output:/app/output" \\
        --network none \\
        "${IMAGE_NAME}:latest" || {
        log_error "Container test failed"
        exit 1
    }
    
    # Check if output was generated
    if [[ -f "test_output/test.json" ]]; then
        log_success "Container test passed - output file generated"
    else
        log_warning "Container ran but no output file found"
    fi
    
    # Cleanup test files
    rm -rf test_input test_output
}

# Run Challenge 1a
run_challenge_1a() {
    local input_dir="${1:-./input}"
    local output_dir="${2:-./output}"
    
    log_info "Running Challenge 1a processing..."
    
    # Check input directory
    if [[ ! -d "$input_dir" ]]; then
        log_error "Input directory $input_dir does not exist"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run the container
    docker run --rm \\
        -v "$(realpath "$input_dir"):/app/input:ro" \\
        -v "$(realpath "$output_dir"):/app/output" \\
        --network none \\
        --memory=16g \\
        --cpus=8 \\
        "${IMAGE_NAME}:latest"
    
    log_success "Challenge 1a processing completed"
    log_info "Output files saved to: $output_dir"
}

# Run Challenge 1b
run_challenge_1b() {
    local work_dir="${1:-./Challenge_1b}"
    
    log_info "Running Challenge 1b processing..."
    
    # Check working directory
    if [[ ! -d "$work_dir" ]]; then
        log_error "Working directory $work_dir does not exist"
        exit 1
    fi
    
    # Run the container
    docker run --rm \\
        -v "$(realpath "$work_dir"):/app" \\
        --network none \\
        --memory=16g \\
        --cpus=8 \\
        "${IMAGE_NAME}:latest" \\
        python unified_processor.py --challenge 1b
    
    log_success "Challenge 1b processing completed"
}

# Run demo
run_demo() {
    log_info "Running system demonstration..."
    
    docker run --rm \\
        --network none \\
        "${IMAGE_NAME}:latest" \\
        python demo.py
    
    log_success "Demo completed"
}

# Clean up Docker resources
cleanup() {
    log_info "Cleaning up Docker resources..."
    
    # Remove old images
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\\t{{.ID}}" | grep -v "latest\\|${VERSION}" | awk '{print $2}' | xargs -r docker rmi
    
    # Clean up unused resources
    docker system prune -f
    
    log_success "Cleanup completed"
}

# Show system information
show_info() {
    log_info "System Information:"
    echo "  Docker Version: $(docker --version)"
    echo "  Platform: $(uname -m)"
    echo "  Available Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "  CPU Cores: $(nproc)"
    
    log_info "Image Information:"
    if docker images "${IMAGE_NAME}:latest" --format "table {{.Repository}}:{{.Tag}}\\t{{.Size}}\\t{{.CreatedAt}}" | grep -q "${IMAGE_NAME}"; then
        docker images "${IMAGE_NAME}:latest" --format "table {{.Repository}}:{{.Tag}}\\t{{.Size}}\\t{{.CreatedAt}}"
    else
        echo "  Image not built yet"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build               Build the Docker image"
    echo "  test                Test the built image"
    echo "  run-1a [input] [output]   Run Challenge 1a (default: ./input ./output)"
    echo "  run-1b [workdir]    Run Challenge 1b (default: ./Challenge_1b)"
    echo "  demo                Run system demonstration"
    echo "  cleanup             Clean up Docker resources"
    echo "  info                Show system information"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run-1a ./my_pdfs ./results"
    echo "  $0 run-1b ./Challenge_1b"
    echo "  $0 demo"
}

# Main script
main() {
    case "${1:-help}" in
        "build")
            check_prerequisites
            build_image
            ;;
        "test")
            test_image
            ;;
        "run-1a")
            run_challenge_1a "$2" "$3"
            ;;
        "run-1b") 
            run_challenge_1b "$2"
            ;;
        "demo")
            run_demo
            ;;
        "cleanup")
            cleanup
            ;;
        "info")
            show_info
            ;;
        "all")
            check_prerequisites
            build_image
            test_image
            log_success "Full deployment completed successfully!"
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"
