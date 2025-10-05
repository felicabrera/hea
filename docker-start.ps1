# Docker Quick Start Script for Windows
# Run this script to start the HEA platform with Docker

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HEA Docker Setup - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker not found. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "  Download: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is running
Write-Host "Checking if Docker is running..." -ForegroundColor Yellow
try {
    docker ps > $null 2>&1
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build and Start Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for existing containers
$existingContainers = docker ps -a --filter "name=hea-" --format "{{.Names}}"
if ($existingContainers) {
    Write-Host "Found existing HEA containers:" -ForegroundColor Yellow
    $existingContainers | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
    $response = Read-Host "Remove existing containers? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Removing existing containers..." -ForegroundColor Yellow
        docker-compose down
        Write-Host "✓ Containers removed" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Building and starting services..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray
Write-Host ""

# Build and start services
docker-compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Services Started Successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access your services at:" -ForegroundColor Cyan
    Write-Host "  🌐 Webapp:      http://localhost:8501" -ForegroundColor White
    Write-Host "  🔌 Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host "  📚 API Docs:    http://localhost:8000/docs" -ForegroundColor White
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Cyan
    Write-Host "  View logs:      docker-compose logs -f" -ForegroundColor Gray
    Write-Host "  Stop services:  docker-compose down" -ForegroundColor Gray
    Write-Host "  Restart:        docker-compose restart" -ForegroundColor Gray
    Write-Host "  Status:         docker-compose ps" -ForegroundColor Gray
    Write-Host ""
    
    # Wait for services to be healthy
    Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Check health
    Write-Host ""
    Write-Host "Service Status:" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host ""
    Write-Host "Opening webapp in browser..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8501"
    
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Error: Failed to start services" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check logs with: docker-compose logs" -ForegroundColor Yellow
    Write-Host "For help, see: docs/DOCKER_SETUP.md" -ForegroundColor Yellow
    exit 1
}
