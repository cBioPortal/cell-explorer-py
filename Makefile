.PHONY: help install dev dev-docker test test-cell2zarr db-migrate db-revision db-seed openapi dev-keys

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	uv sync

dev: ## Start API dev server with hot-reload
	uv run --project packages/api uvicorn cell_explorer_api.main:app --reload --port 8000

dev-docker: ## Start API via docker-compose (serves frontend from dist)
	docker compose up

test: ## Run API tests
	uv run --project packages/api pytest packages/api/tests/ -v

test-cell2zarr: ## Run cell2zarr tests
	uv run --project packages/cell2zarr pytest packages/cell2zarr/tests/ -v

db-migrate: ## Run Alembic migrations (upgrade to head)
	uv run --project packages/api alembic -c packages/api/alembic.ini upgrade head

db-revision: ## Generate a new Alembic migration (usage: make db-revision msg="description")
	uv run --project packages/api alembic -c packages/api/alembic.ini revision --autogenerate -m "$(msg)"

db-seed: ## Seed the database with sample dev data
	uv run --project packages/api python -m cell_explorer_api.db.seed

openapi: ## Regenerate OpenAPI spec
	uv run --project packages/api python -m cell_explorer_api.export_openapi > openapi.json

dev-keys: ## Generate RSA key pair for local dev
	mkdir -p dev-keys
	openssl genrsa -out dev-keys/private.pem 2048
	openssl rsa -in dev-keys/private.pem -pubout -out dev-keys/public.pem
	@echo "Keys generated in dev-keys/. Add DATASOURCE_LOCAL_PRIVATE_KEY_FILE=/keys/private.pem to .env"
