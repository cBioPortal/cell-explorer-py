.PHONY: help install dev dev-docker test test-cell2zarr test-zarr-auth-proxy test-zarr-access test-all db-migrate db-revision db-seed seed-spectrum openapi generate-app-config dev-keys chat-login chat-logout chat-datasets chat-ask chat-ask-traced chat-repl

# chat-* targets auto-source .env so the CLI picks up KEYCLOAK_*, CELL_EXPLORER_API_URL, etc.
WITH_ENV = set -a; [ -f .env ] && . ./.env; set +a;

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

seed-spectrum: ## Register the public MSK SPECTRUM TME 2022 dataset (idempotent)
	uv run --project packages/api python scripts/seed_spectrum.py

openapi: ## Regenerate OpenAPI spec
	uv run --project packages/api python -m cell_explorer_api.export_openapi > openapi.json

generate-app-config: ## Regenerate AppConfig Pydantic model from JSON Schema artifact
	uv run --project packages/cell-explorer-agent datamodel-codegen \
	  --input packages/cell-explorer-agent/schema/app_config.schema.json \
	  --input-file-type jsonschema \
	  --output packages/cell-explorer-agent/src/cell_explorer_agent/schema/app_config.py \
	  --target-python-version 3.12 \
	  --output-model-type pydantic_v2.BaseModel \
	  --use-standard-collections \
	  --use-union-operator \
	  --enum-field-as-literal all \
	  --field-extra-keys description \
	  --disable-timestamp

test-zarr-auth-proxy: ## Run zarr-auth-proxy tests
	uv run --project packages/zarr-auth-proxy pytest packages/zarr-auth-proxy/tests/ -v

test-zarr-access: ## Run zarr-access tests
	uv run --project packages/zarr-access pytest packages/zarr-access/tests/ -v

test-all: ## Run all tests across all packages
	uv run --project packages/api pytest packages/api/tests/ -v
	uv run --project packages/cell2zarr pytest packages/cell2zarr/tests/ -v
	uv run --project packages/zarr-auth-proxy pytest packages/zarr-auth-proxy/tests/ -v
	uv run --project packages/zarr-access pytest packages/zarr-access/tests/ -v

dev-keys: ## Generate RSA key pair for local dev
	mkdir -p dev-keys
	openssl genrsa -out dev-keys/private.pem 2048
	openssl rsa -in dev-keys/private.pem -pubout -out dev-keys/public.pem
	@echo "Keys generated in dev-keys/. Add DATASOURCE_LOCAL_PRIVATE_KEY_FILE=/keys/private.pem to .env"

chat-login: ## Open browser-OAuth login flow (auto-loads .env)
	@$(WITH_ENV) uv run cell-explorer-chat login

chat-logout: ## Delete local CLI auth.json
	@$(WITH_ENV) uv run cell-explorer-chat logout

chat-datasets: ## List datasets the authenticated user can access
	@$(WITH_ENV) uv run cell-explorer-chat datasets

chat-ask: ## Ask one question (usage: make chat-ask SLUG=spectrum Q="how many cells?")
	@$(WITH_ENV) uv run cell-explorer-chat ask $(SLUG) "$(Q)"

chat-ask-traced: ## Ask + write a zarr HTTP trace (usage: make chat-ask-traced SLUG=... Q="..." [TRACE=path.jsonl])
	@trace="$(or $(TRACE),trace-$$(date +%Y%m%d-%H%M%S).jsonl)"; \
	echo "tracing to $$trace"; \
	$(WITH_ENV) ZARR_ACCESS_TRACE=1 ZARR_ACCESS_TRACE_FILE="$$trace" /usr/bin/time -p uv run cell-explorer-chat ask $(SLUG) "$(Q)"

chat-repl: ## Multi-turn chat REPL (usage: make chat-repl SLUG=spectrum)
	@$(WITH_ENV) uv run cell-explorer-chat repl $(SLUG)
