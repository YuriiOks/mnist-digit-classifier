# Makefile for MNIST Digit Classifier

# Variables
DOCKER_COMPOSE = docker-compose
APP_NAME = mnist-digit-classifier

# Default target
.PHONY: help
help:
	@echo "MNIST Digit Classifier Docker Management"
	@echo ""
	@echo "Usage:"
	@echo "  make build-all          Build all Docker images"
	@echo "  make run                Run all containers"
	@echo "  make stop               Stop all containers"
	@echo "  make clean              Remove all containers and volumes"
	@echo ""
	@echo "Individual Components:"
	@echo "  make build-model        Build model service image"
	@echo "  make build-web          Build web service image"
	@echo "  make run-model          Run model service only"
	@echo "  make run-web            Run web service only"
	@echo "  make run-db             Run database only"
	@echo ""
	@echo "Development:"
	@echo "  make logs               Show logs from all services"
	@echo "  make logs-web           Show logs from web service"
	@echo "  make logs-model         Show logs from model service"
	@echo "  make logs-db            Show logs from database"
	@echo "  make status             Show status of all containers"
	@echo "  make train-model        Train a new model"

# Build all images
.PHONY: build-all
build-all:
	$(DOCKER_COMPOSE) build

# Run all containers
.PHONY: run
run:
	$(DOCKER_COMPOSE) up -d

# Run all containers and show logs
.PHONY: run-dev
run-dev:
	$(DOCKER_COMPOSE) up

# Stop all containers
.PHONY: stop
stop:
	$(DOCKER_COMPOSE) stop

# Remove all containers and volumes
.PHONY: clean
clean:
	$(DOCKER_COMPOSE) down -v

# Build individual images
.PHONY: build-model
build-model:
	$(DOCKER_COMPOSE) build model

.PHONY: build-web
build-web:
	$(DOCKER_COMPOSE) build web

# Run individual containers
.PHONY: run-model
run-model:
	$(DOCKER_COMPOSE) up -d model

.PHONY: run-web
run-web:
	$(DOCKER_COMPOSE) up -d web db model

.PHONY: run-db
run-db:
	$(DOCKER_COMPOSE) up -d db

# Show logs
.PHONY: logs
logs:
	$(DOCKER_COMPOSE) logs -f

.PHONY: logs-web
logs-web:
	$(DOCKER_COMPOSE) logs -f web

.PHONY: logs-model
logs-model:
	$(DOCKER_COMPOSE) logs -f model

.PHONY: logs-db
logs-db:
	$(DOCKER_COMPOSE) logs -f db

# Show status of containers
.PHONY: status
status:
	$(DOCKER_COMPOSE) ps

# Train a new model
.PHONY: train-model
train-model:
	@echo "Running model training script..."
	@if [ -f model/train.py ]; then \
		$(DOCKER_COMPOSE) run --rm model python train.py; \
	else \
		echo "Error: model/train.py not found"; \
	fi

# Enter a shell in a container
.PHONY: shell-web
shell-web:
	$(DOCKER_COMPOSE) exec web bash

.PHONY: shell-model
shell-model:
	$(DOCKER_COMPOSE) exec model bash

.PHONY: shell-db
shell-db:
	$(DOCKER_COMPOSE) exec db bash 