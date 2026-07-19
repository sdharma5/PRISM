#!/usr/bin/env bash
# Start the frontend without colliding with anyone else using this checkout.
#
# Several of us share this directory on a cluster filesystem. Three things
# collide when two people run the app at once:
#
#   .next       -> handled by next.config.js, which makes distDir per-user
#   port 3000   -> handled here, by deriving a per-user port
#   the API URL -> handled here, by pointing at that user's own backend
#
# Real environment variables take precedence over .env.local in Next, so the
# values exported here win without anyone editing the shared .env file.
#
# Usage:
#   bash scripts/dev.sh                  # per-user defaults
#   PORT=3005 API_PORT=8005 bash scripts/dev.sh
#   MODE=mock bash scripts/dev.sh        # no backend needed

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

USER_NAME="${USER:-shared}"

# Deterministic per-user offset, so the same person always gets the same ports
# and two different people almost never get the same pair. Stable across
# restarts matters: a port that moves every launch breaks your SSH tunnel.
OFFSET=$(( $(printf '%s' "$USER_NAME" | cksum | cut -d' ' -f1) % 40 ))
PORT="${PORT:-$(( 3000 + OFFSET ))}"
API_PORT="${API_PORT:-$(( 8000 + OFFSET ))}"
MODE="${MODE:-http}"

export NEXT_DIST_DIR="${NEXT_DIST_DIR:-.next-$USER_NAME}"
export NEXT_PUBLIC_PRISM_API_MODE="$MODE"
export NEXT_PUBLIC_PRISM_API_URL="http://localhost:${API_PORT}"

TOOLCHAIN="$(cd "$APP_DIR/../.." && pwd)/.toolchain/node/bin"
[ -d "$TOOLCHAIN" ] && export PATH="$TOOLCHAIN:$PATH"

cat <<INFO
Starting PRISM frontend
  user       : $USER_NAME
  frontend   : http://localhost:$PORT
  API        : $NEXT_PUBLIC_PRISM_API_URL  (mode=$MODE)
  build dir  : $NEXT_DIST_DIR

Tunnel from your laptop with BOTH ports -- the browser calls the API directly:
  ssh -N -L $PORT:\$(hostname -f):$PORT -L $API_PORT:\$(hostname -f):$API_PORT \\
      $USER_NAME@<login-node>
INFO

if [ "$MODE" = "http" ] && ! curl -sf -o /dev/null "http://127.0.0.1:${API_PORT}/api/v1/health" 2>/dev/null; then
  cat <<WARN

  Note: nothing is listening on ${API_PORT}. Start your backend with:
    cd ../../Hack-Nation && .venv/bin/uvicorn apps.api.main:app --port ${API_PORT}
  Or run this script with MODE=mock to use the bundled demo patients instead.
WARN
fi

exec npx next dev -p "$PORT"
