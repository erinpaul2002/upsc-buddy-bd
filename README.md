# UPSC Buddy Backend

FastAPI backend for OMR answer extraction, answer-key OCR extraction, and result evaluation.

## Tech Stack

- FastAPI + Uvicorn
- OpenCV + NumPy + imutils
- RapidOCR (onnxruntime)
- pdf2image (first-page PDF support)

## Run Locally

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API base URL: `http://localhost:8000`

## API Routes

### Health

- `GET /health`

### Extract OMR Answers

- `POST /api/v1/omr/extract`
- `multipart/form-data`
	- `omr_file` (required, `jpg|jpeg|png|pdf`)
	- `total_questions` (optional, default `100`)

Response includes:
- `answers` (array)
- `detected_bubbles` (int)

### Extract Answer Key

- `POST /api/v1/answer-key/extract`
- `multipart/form-data`
	- `key_file` (required, `jpg|jpeg|png`)
	- `total_questions` (optional, default `100`)

Response includes:
- `answers` (array)
- `ocr_analysis` (phase timing breakdown for OCR pipeline)
- `request_analysis` (API-level timing: upload read/decode/process)
- `analysis_log` (copy-paste ready plain-text performance report)

### Evaluate Submission

- `POST /api/v1/evaluate`
- `application/json`
	- `student_answers` (required, array of extracted OMR answers)
	- `correct_answers` (required, array of extracted answer-key answers)
	- `total_questions` (optional, default `100`)

Response includes:
- `evaluation.score`
- `evaluation.stats`
- `evaluation.results` (per-question status)

## OCR Multiprocessing (Optional)

Answer-key cell OCR can run with multiprocessing workers.

```bash
OCR_MP_WORKERS=2
```

- Default is `1` (single-process).
- Increase only after benchmarking on your deployment machine.

## CORS

Set allowed origins with env var:

```bash
CORS_ORIGINS=http://localhost:3000,https://your-frontend-domain.com
```

If not set, defaults to `*` for development.

## Deploy on Render (Docker)

`Dockerfile` is included for backend-only deployment (no docker-compose).

- Exposes port `8000`
- Installs `poppler-utils` and OpenCV runtime libs

For Render Docker service, point to this folder and use the provided `Dockerfile`.

## Render Blueprint

A minimal Render blueprint is included at `render.yaml` with a Docker-only service definition:

- `upsc-buddy-bd-docker` (Docker runtime)

Use this service directly for Render Docker deployment.