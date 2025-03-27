#!/bin/bash

echo "Starting MNIST Digit Classifier Application..."

# Build all images
echo "Building Docker images..."
docker-compose build

# Run the application
echo "Starting containers..."
docker-compose up -d

# Check if containers are running
echo "Checking container status..."
docker-compose ps

echo ""
echo "Application is running!"
echo "Web interface: http://localhost:8501"
echo ""
echo "Use 'docker-compose logs -f' to view logs"
echo "Use 'docker-compose down' to stop and remove containers" 