#!/bin/bash

echo "======================================"
echo "Fleet Backend API"
echo "Starting Development Server"
echo "======================================"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null
then
    echo "Error: docker-compose is not installed"
    echo "Please install Docker and Docker Compose first"
    exit 1
fi

echo "Starting backend service with hot reload..."
echo ""

docker-compose -f docker-compose.dev.yml up --build

echo ""
echo "======================================"
echo "Backend API stopped"
echo "======================================"
