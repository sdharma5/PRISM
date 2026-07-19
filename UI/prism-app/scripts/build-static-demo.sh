#!/usr/bin/env bash
# Build a self-contained static demo — no backend, no SSH tunnel.
#
# Produces a folder of plain HTML/JS that runs from any static file server. It
# is built in mock mode, so the three generated demo patients are bundled into
# the JavaScript and no API is contacted at run time.
#
# What is deliberately NOT in the export:
#
#   /intake, /inputs/voice   need the live API (registry-driven fields; Whisper)
#   /care, /recommendations  need server route handlers, which cannot be
#                            statically exported
#
# They are removed from the *copy* rather than stubbed, so the export only
# contains pages that work. The working tree is untouched.
#
# Usage:
#   bash scripts/build-static-demo.sh            # -> ./static-demo/
#   bash scripts/build-static-demo.sh /some/dir  # -> /some/dir

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$APP_DIR/static-demo}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "==> Copying project to a scratch build dir"
cd "$APP_DIR"
tar --exclude=node_modules --exclude=.next --exclude=static-demo -cf - . \
  | (mkdir -p "$WORK_DIR/app" && tar -xf - -C "$WORK_DIR/app")
ln -s "$APP_DIR/node_modules" "$WORK_DIR/app/node_modules"

cd "$WORK_DIR/app"

echo "==> Removing routes that require a server"
rm -rf src/app/api src/app/care src/app/recommendations src/app/intake src/app/inputs/voice

echo "==> Forcing mock mode and static output"
# .env.local outranks .env.production in Next's precedence order, so every local
# override is removed and the value is set in the build environment instead,
# where nothing can shadow it.
rm -f .env.local .env.development.local .env.production.local .env.development
cat > .env.production <<'ENV'
NEXT_PUBLIC_PRISM_API_MODE=mock
ENV
export NEXT_PUBLIC_PRISM_API_MODE=mock

cat > next.config.js <<'CONFIG'
/** @type {import('next').NextConfig} */
module.exports = {
  output: 'export',
  // Relative asset paths so the export also works when opened from a
  // subdirectory or served from a path prefix.
  images: { unoptimized: true },
  trailingSlash: true,
}
CONFIG

echo "==> Building (mode=$NEXT_PUBLIC_PRISM_API_MODE)"
NEXT_PUBLIC_PRISM_API_MODE=mock npx next build

echo "==> Collecting output"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp -r out/* "$OUT_DIR"/

cat > "$OUT_DIR/README.txt" <<'TXT'
PRISM static demo
=================

Self-contained. No backend required.

Run it:

    cd this-folder
    python3 -m http.server 8080

then open http://localhost:8080/overview/

Opening the .html files directly with file:// will not work — the app loads
JavaScript by absolute path, which browsers block on the file:// protocol.
Any static server works; python3 is just the one everyone already has.

Data: the three generated demo patients (Sarah, Priya, Maya). These are real
model outputs, produced by running the actual inference pipeline, not
hand-written values.

Not included: /intake and /inputs/voice (need the live API), /care and
/recommendations (need server routes).
TXT

echo
echo "==> Done: $OUT_DIR"
echo "    cd '$OUT_DIR' && python3 -m http.server 8080"
echo "    then open http://localhost:8080/overview/"
