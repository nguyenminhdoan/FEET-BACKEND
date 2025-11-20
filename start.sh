#!/bin/bash

echo "======================================"
echo "Fleet Backend API"
echo "Starting Production Server"
echo "======================================"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null
then
    echo "Error: docker-compose is not installed"
    echo "Please install Docker and Docker Compose first"
    exit 1
fi

echo "Building and starting backend service..."
echo ""

docker-compose up -d --build

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
echo "  docker-compose logs -f"
echo ""
echo "To stop the service:"
echo "  docker-compose down"
echo ""
