#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LLM_PROVIDER="${LLM_PROVIDER:-local_stub}"
LLM_MODEL="${LLM_MODEL:-local-restaurant-smoke}"
RAG_ENABLED="${RAG_ENABLED:-false}"
STRICT_DEPENDENCY_HEALTH="${STRICT_DEPENDENCY_HEALTH:-false}"

if ! command -v cloudflared >/dev/null 2>&1; then
  cat >&2 <<'MSG'
cloudflared is required to expose your local server to Vapi.
Install it first:
  macOS:   brew install cloudflared
  Linux:   download from https://github.com/cloudflare/cloudflared/releases
  Windows: winget install --id Cloudflare.cloudflared
MSG
  exit 1
fi

cleanup() {
  if [[ -n "${APP_PID:-}" ]]; then
    kill "${APP_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export LLM_PROVIDER LLM_MODEL RAG_ENABLED STRICT_DEPENDENCY_HEALTH

uvicorn app.main:app --host "${HOST}" --port "${PORT}" &
APP_PID="$!"

cat <<MSG

Local Vapi test backend is starting on http://localhost:${PORT}
Opening a temporary HTTPS Cloudflare Tunnel...

IMPORTANT: Put the HTTPS base URL printed below in Vapi Custom LLM.
Do NOT put localhost in Vapi. Vapi will append /chat/completions automatically.

MSG

cloudflared tunnel --url "http://localhost:${PORT}" --no-autoupdate --edge-ip-version 4
