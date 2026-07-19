# PRISM — Patient-Facing Frontend

Next.js 14 web application that lets patients and clinicians explore a PRISM phenotype profile.

---

## What this is

The frontend connects to the PRISM backend (`Hack-Nation/`) and presents evidence, phenotype domains, care recommendations, and data gaps in a structured, uncertainty-aware UI.

**Not a medical device.** All outputs are research-grade and must not be used for clinical decisions.

---

## Pages

| Route | Purpose |
|---|---|
| `/` | Landing page |
| `/onboarding` | Account setup and context collection |
| `/intake` | Upload labs, documents, questionnaire, wearable data |
| `/overview` | Phenotype profile summary |
| `/care` | Personalized recommendations |
| `/chat` | Ask questions about your profile |
| `/cycle` | Cycle and hormonal timeline |
| `/dashboard` | At-a-glance evidence status |
| `/timeline` | Longitudinal event view |
| `/review` | Evidence and confirmation queue |
| `/research` | How PRISM works |
| `/profile` | Patient profile settings |
| `/settings` | App settings |

---

## Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Recharts
- **State**: Zustand
- **Language**: TypeScript

---

## Setup

```bash
npm install
npm run dev       # http://localhost:3000
```

**Generate types and demo fixtures** (requires the Python backend):

```bash
npm run gen:types   # exports OpenAPI types from Hack-Nation
npm run gen:demo    # regenerates demo patient fixtures
```

---

## Demo mode

Static demo patients live in `src/lib/demo/`. The app can run entirely from these fixtures without a running backend — useful for UI development and the hackathon demo.

---

## Connection to the backend

The API client in `src/lib/apiClient.ts` points to the FastAPI server in `Hack-Nation/apps/api/`. Set `NEXT_PUBLIC_API_URL` in a `.env.local` file to override the default.
