# VoxGuard frontend

React 19 + Vite + Tailwind 4.

```bash
cd frontend
npm install
cp .env.example .env     # set VITE_API_URL if the backend is not on :8000
npm run dev              # http://localhost:5173
```

Backend (separate terminal): `cd backend && uvicorn app.main:app --reload`

The UI calls `GET /health`, `POST /upload`, then `POST /predict/{sample_id}`.
`/predict` is not in the backend yet; it should return `{ "label": "real" | "fake", "confidence": 0.0-1.0 }`.
