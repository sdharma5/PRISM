# Launching PRISM Locally

PRISM has two parts that run as separate processes:

| Part | Location | Stack | Default port |
|---|---|---|---|
| **Frontend** | `UI/prism-app/` | Next.js 14 (TypeScript) | `3000` |
| **Backend API** | `Hack-Nation/` | FastAPI + uvicorn (Python 3.11+) | `8000` |

You have two options:

- **Fast path (frontend only, no backend):** the UI ships with committed demo fixtures and runs entirely from them. Best for exploring the interface. → do **Step 1** only.
- **Full stack (frontend + live backend):** run the FastAPI service and point the UI at it. → do **Step 1 and Step 2**.

---

## Prerequisites

- **Node.js 18+** and npm (for the frontend)
- **Python 3.11+** (for the backend, full-stack option only)
- **git**

---

## 0. Clone the repo

```bash
git clone https://github.com/sdharma5/PRISM.git
cd PRISM
```

---

## 1. Frontend (Next.js)

```bash
cd UI/prism-app
npm install
cp .env.example .env.local     # defaults to mock mode — no backend needed
npm run dev
```

Open **http://localhost:3000**.

Data source is controlled by `.env.local`:

```bash
NEXT_PUBLIC_PRISM_API_MODE=mock                    # committed demo fixtures, no backend
# NEXT_PUBLIC_PRISM_API_MODE=http                  # call the live FastAPI backend
# NEXT_PUBLIC_PRISM_API_URL=http://localhost:8000  # backend URL (used only in http mode)
```

Leave it as `mock` to explore the UI immediately. Switch to `http` after you have the backend running (Step 2).

**Frontend scripts** (run from `UI/prism-app/`):

```bash
npm run dev        # dev server with hot reload → http://localhost:3000
npm run build      # production build
npm run start      # serve the production build on port 3000
npm run lint       # eslint
npm run typecheck  # tsc --noEmit
```

---

## 2. Backend API (FastAPI) — full-stack option only

From the repo root:

```bash
cd Hack-Nation

# Preferred: uses uv if available, falls back to a venv + pip
make install

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api]"     # FastAPI + uvicorn (minimum to run the API)
# pip install -e ".[dev]"   # everything, incl. optional imaging/speech/documents + tests
```

Start the server:

```bash
python3 -m uvicorn apps.api.main:app --reload --port 8000
```

The API is now at **http://localhost:8000** (interactive docs at **http://localhost:8000/docs**).

> Startup loads all model encoders once. If the core (static) branch can't load, startup deliberately fails rather than serving degraded results.

Heavy dependencies are optional extras and every module degrades gracefully without them:

```bash
pip install -e ".[imaging]"     # PyTorch + pydicom + scikit-image (ultrasound)
pip install -e ".[speech]"      # Whisper (voice input)
pip install -e ".[documents]"   # pdfplumber (lab-report PDFs)
```

### Wire the frontend to the backend

With the backend running, edit `UI/prism-app/.env.local`:

```bash
NEXT_PUBLIC_PRISM_API_MODE=http
NEXT_PUBLIC_PRISM_API_URL=http://localhost:8000
```

Restart `npm run dev`. The UI now pulls live data from your local API.

---

## Typical full-stack workflow (two terminals)

```bash
# Terminal 1 — backend
cd PRISM/Hack-Nation
source .venv/bin/activate
python3 -m uvicorn apps.api.main:app --reload --port 8000

# Terminal 2 — frontend (with .env.local set to http mode)
cd PRISM/UI/prism-app
npm run dev
```

Then open **http://localhost:3000**.

---

## Troubleshooting

- **Port already in use** — change the port: `npm run dev -- -p 3001` or `--port 8001` for uvicorn (update `NEXT_PUBLIC_PRISM_API_URL` to match).
- **`uv` not found** — `make install` automatically falls back to `python3 -m venv .venv && pip install -e ".[dev]"`.
- **UI shows no live data** — confirm `NEXT_PUBLIC_PRISM_API_MODE=http`, the backend is up at the URL in `NEXT_PUBLIC_PRISM_API_URL`, and you restarted the dev server after editing `.env.local`.
- **`.env` for the backend does not auto-load** — nothing reads it automatically. Export it yourself if you need it: `set -a; source .env; set +a`. Not required for a basic API run.

---

> **Not a medical device.** PRISM is a research artifact and does not diagnose any condition. All outputs are research-grade and must not be used for clinical decisions.
