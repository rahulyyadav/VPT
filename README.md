# VITGPT – A Minimal ChatGPT‑style Interface for the VIT Dataset

VITGPT is a lightweight ChatGPT‑style web application that serves a TinyGPT model trained on a curated VIT Q&A dataset. The frontend is built with Next.js 14 and TypeScript; model inference is handled by a Python/PyTorch program.

This repository contains:

- A modern chat UI (Next.js App Router)
- An API route that bridges the UI to model inference
- A Python implementation of a TinyGPT variant and an inference entrypoint

Note: Large model artifacts are intentionally excluded from git. See “Deployment” for production guidance.

## Contents

- Features
- Architecture
- Local Development
- Inference Service (Python)
- Environment Variables
- Deployment (Vercel + external inference)
- Dataset & Model Notes
- License

## Features

- Clean, responsive chat interface (App Router, CSS modules)
- Simple API contract between UI and inference
- TinyGPT implementation in pure PyTorch for clarity
- Deterministic generation settings with optional top‑k

## Architecture

- `app/page.tsx`: chat UI
- `app/api/chat/route.ts`: POST `/api/chat` → calls Python inference
- `model_inference.py`: loads the trained checkpoint and generates responses
- `tinygpt_model.py`: TinyGPT model (attention blocks, generation loop)

Request flow:

1. UI sends `{ message: string }` to `/api/chat`
2. The API route executes `model_inference.py` with the message
3. Python loads the model checkpoint, generates text, returns `{ response }`

## Local Development

Prerequisites: Node 18+, Python 3.10+, pip, virtualenv (recommended).

1. Install Node deps

```bash
npm install
```

2. (Recommended) Create and activate a Python virtual env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Place your trained checkpoint at the project root as `vpt_model_epoch_100.pt`

4. Start the Next.js dev server

```bash
npm run dev
```

Open `http://localhost:3000`.

## Inference Service (Python)

The default API route launches `model_inference.py` as a subprocess:

- Expects `vpt_model_epoch_100.pt` at project root
- Uses CPU by default; uses CUDA if available
- I/O is JSON over stdout

If you prefer a long‑running service (recommended for production), expose the model via a small HTTP server (e.g., FastAPI) and have the Next.js API call it via `fetch`. This avoids cold‑starting Python for each request.

Example FastAPI sketch (not included in repo):

```python
from fastapi import FastAPI
from pydantic import BaseModel
from model_inference import load_model, generate_response

app = FastAPI()
model, stoi, itos, device = load_model()

class ChatIn(BaseModel):
    message: str

@app.post("/infer")
def infer(inp: ChatIn):
    if model is None:
        return {"response": "Model not available"}
    return {"response": generate_response(model, stoi, itos, device, inp.message)}
```

## Environment Variables

If you deploy inference separately, set in the Next.js app:

- `INFERENCE_API_URL` – URL of your inference endpoint (e.g., `https://your-api.example.com/infer`)

Update `app/api/chat/route.ts` to use `fetch(INFERENCE_API_URL, { method: 'POST', body: JSON.stringify({ message }) })` instead of spawning Python. This keeps the frontend Vercel‑friendly.

## Deployment

### Will this run on Vercel as‑is?

Not reliably. The current implementation spawns a Python process and expects a large `.pt` file present on the server. Vercel Serverless Functions run on a Node.js runtime for this route, with tight size and execution limits; shipping PyTorch and a large checkpoint is not supported. Also, we removed the large checkpoint from git (by design).

### Recommended production setup

1. Host inference externally (Railway, Render, Fly.io, EC2, GCP, etc.). Containerize a small FastAPI app that loads `vpt_model_epoch_100.pt` once at startup.
2. Deploy the Next.js frontend to Vercel. Point the API route to the external inference URL via `INFERENCE_API_URL`.
3. For privacy or latency, place inference close to your users or use GPU instances if needed.

### Optional: Git LFS

If you must keep the checkpoint in your repo, use Git LFS and a host that supports large files. This is not recommended for Vercel Serverless.

## Dataset & Model Notes

- Dataset: VIT Q&A (see `vit_clean_qa.txt` for sample content)
- Model: TinyGPT (multi‑head attention blocks, causal masking, simple sampler)
- Inference defaults: `max_new_tokens=200`, `temperature=0.1`, optional `top_k`

## License

MIT
