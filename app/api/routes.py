import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.processing import (
    decode_uploaded_file,
    evaluate_answers,
    process_answer_key,
    process_omr,
)

router = APIRouter()


class EvaluateRequest(BaseModel):
    student_answers: list[str] = Field(default_factory=list)
    correct_answers: list[str] = Field(default_factory=list)
    total_questions: int = 100


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/omr/extract")
async def extract_omr_answers(
    omr_file: UploadFile = File(...),
    total_questions: int = Form(100),
) -> dict:
    try:
        raw = await omr_file.read()
        image = decode_uploaded_file(raw, omr_file.content_type)
        answers, bubble_count = process_omr(image, total_questions=total_questions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OMR processing failed: {exc}") from exc

    return {
        "answers": answers,
        "total_questions": total_questions,
        "detected_bubbles": bubble_count,
        "source_filename": omr_file.filename,
    }


@router.post("/api/v1/answer-key/extract")
async def extract_key_answers(
    key_file: UploadFile = File(...),
    total_questions: int = Form(100),
) -> dict:
    request_started = time.perf_counter()
    try:
        phase_started = time.perf_counter()
        raw = await key_file.read()
        read_ms = round((time.perf_counter() - phase_started) * 1000.0, 2)

        phase_started = time.perf_counter()
        image = decode_uploaded_file(raw, key_file.content_type)
        decode_ms = round((time.perf_counter() - phase_started) * 1000.0, 2)

        phase_started = time.perf_counter()
        answers, ocr_analysis, ocr_analysis_log = process_answer_key(
            image, total_questions=total_questions
        )
        process_ms = round((time.perf_counter() - phase_started) * 1000.0, 2)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Answer key processing failed: {exc}"
        ) from exc

    total_ms = round((time.perf_counter() - request_started) * 1000.0, 2)
    request_analysis = {
        "total_ms": total_ms,
        "phases": [
            {"name": "request_read_upload", "ms": read_ms},
            {"name": "request_decode_file", "ms": decode_ms},
            {"name": "request_ocr_processing", "ms": process_ms},
        ],
    }

    request_log = "\n".join(
        [
            "API Request Timing",
            "==================",
            f"Total time: {total_ms} ms",
            f"- request_read_upload: {read_ms} ms",
            f"- request_decode_file: {decode_ms} ms",
            f"- request_ocr_processing: {process_ms} ms",
        ]
    )

    return {
        "answers": answers,
        "total_questions": total_questions,
        "source_filename": key_file.filename,
        "ocr_analysis": ocr_analysis,
        "request_analysis": request_analysis,
        "analysis_log": f"{request_log}\n\n{ocr_analysis_log}",
    }


@router.post("/api/v1/evaluate")
async def evaluate_submission(payload: EvaluateRequest) -> dict:
    try:
        evaluation = evaluate_answers(
            student_answers=payload.student_answers,
            correct_answers=payload.correct_answers,
            total_questions=payload.total_questions,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {exc}") from exc

    return {"evaluation": evaluation}
