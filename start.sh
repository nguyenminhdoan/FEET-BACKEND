#!/bin/bash

echo "======================================"
echo "Fleet Backend API"
echo "Starting Production Server"
echo "======================================"
echo ""

# Check if docker compose is available (v2 or v1)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "Error: docker-compose is not installed"
    echo "Please install Docker and Docker Compose first"
    exit 1
fi

echo "Building and starting backend service..."
echo ""

$DOCKER_COMPOSE up -d --build

echo ""
echo "======================================"
echo "Backend API started successfully!"
echo "======================================"
echo ""
echo "Access the API at:"
echo "  - API Endpoint: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/api/health"
echo ""
echo "To view logs:"
echo "  $DOCKER_COMPOSE logs -f"
echo ""
echo "To stop the service:"
echo "  $DOCKER_COMPOSE down"
echo ""
