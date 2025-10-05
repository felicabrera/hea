#!/bin/bash
# Docker Quick Start Script for Linux/Mac
# Run this script to start the HEA platform with Docker

echo "========================================"
echo "  HEA Docker Setup - Quick Start"
echo "========================================"
echo ""

# Check if Docker is installed
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "✗ Docker not found. Please install Docker first."
    echo "  Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✓ Docker found: $(docker --version)"

# Check if Docker is running
echo "Checking if Docker is running..."
if ! docker ps &> /dev/null; then
    echo "✗ Docker is not running. Please start Docker."
    exit 1
fi
echo "✓ Docker is running"

echo ""
echo "========================================"
echo "  Build and Start Services"
echo "========================================"
echo ""

# Check for existing containers
if [ "$(docker ps -a --filter 'name=hea-' --format '{{.Names}}')" ]; then
    echo "Found existing HEA containers:"
    docker ps -a --filter 'name=hea-' --format '  - {{.Names}}'
    echo ""
    read -p "Remove existing containers? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing containers..."
        docker-compose down
        echo "✓ Containers removed"
    fi
fi

echo ""
echo "Building and starting services..."
echo "This may take a few minutes on first run..."
echo ""

# Build and start services
docker-compose up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Services Started Successfully!"
    echo "========================================"
    echo ""
    echo "Access your services at:"
    echo "  🌐 Webapp:      http://localhost:8501"
    echo "  🔌 Backend API: http://localhost:8000"
    echo "  📚 API Docs:    http://localhost:8000/docs"
    echo ""
    echo "Useful commands:"
    echo "  View logs:      docker-compose logs -f"
    echo "  Stop services:  docker-compose down"
    echo "  Restart:        docker-compose restart"
    echo "  Status:         docker-compose ps"
    echo ""
    
    # Wait for services to be healthy
    echo "Waiting for services to be ready..."
    sleep 5
    
    # Check health
    echo ""
    echo "Service Status:"
    docker-compose ps
    
    echo ""
    echo "Opening webapp in browser..."
    sleep 3
    
    # Open browser (Linux/Mac)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:8501
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:8501 2>/dev/null || echo "Please open http://localhost:8501 in your browser"
    fi
    
else
    echo ""
    echo "========================================"
    echo "  Error: Failed to start services"
    echo "========================================"
    echo ""
    echo "Check logs with: docker-compose logs"
    echo "For help, see: docs/DOCKER_SETUP.md"
    exit 1
fi
