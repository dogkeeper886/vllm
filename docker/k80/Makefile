# vLLM K80 Build System
# Follows the dogkeeper886/ollama37 Makefile pattern.
#
# Usage:
#   make build-builder    # Build the builder image (~120 min first time)
#   make build-runtime    # Build runtime from GitHub clone
#   make build-local      # Build runtime from local source
#   make build            # Build builder + runtime (GitHub)
#   make build-all-local  # Build builder + runtime (local)
#   make run              # Run with docker compose
#   make clean            # Remove images

include .env

BUILDER_IMAGE  ?= vllm37-builder
RUNTIME_IMAGE  ?= vllm37
RUNTIME_LOCAL  ?= vllm37-local
REPO_ROOT      := $(shell git rev-parse --show-toplevel 2>/dev/null || echo ../..)

.PHONY: help build-builder build-runtime build-local build build-all-local run stop logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build-builder: ## Build the builder image (CUDA 11.4 + GCC 10 + Python 3.10 + PyTorch 2.0.1)
	docker build \
		-t $(BUILDER_IMAGE) \
		-f builder/Dockerfile \
		--build-arg JOBS=$(JOBS) \
		.

build-runtime: ## Build runtime image from GitHub clone
	docker build \
		-t $(RUNTIME_IMAGE) \
		-f runtime/Dockerfile \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg VLLM_REPO=$(VLLM_REPO) \
		--build-arg VLLM_BRANCH=$(VLLM_BRANCH) \
		--build-arg MAX_JOBS=$(JOBS) \
		.

build-local: ## Build runtime image from local source
	docker build \
		-t $(RUNTIME_LOCAL) \
		-f runtime/Dockerfile.local \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg MAX_JOBS=$(JOBS) \
		$(REPO_ROOT)

build: build-builder build-runtime ## Build builder + runtime (GitHub clone)

build-all-local: build-builder build-local ## Build builder + runtime (local source)

run: ## Start vLLM server via docker compose
	docker compose up -d

stop: ## Stop vLLM server
	docker compose down

logs: ## Show vLLM server logs
	docker compose logs -f

clean: ## Remove builder and runtime images
	-docker rmi $(RUNTIME_IMAGE) $(RUNTIME_LOCAL) $(BUILDER_IMAGE) 2>/dev/null
	@echo "Cleaned up images."
