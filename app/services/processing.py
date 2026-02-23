from __future__ import annotations

import io

import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image

# Register HEIF/HEIC opener with Pillow (must be done before first use)
import pillow_heif  # noqa: F401
pillow_heif.register_heif_opener()

from app.services.answer_key_utils import extract_answer_key_with_diagnostics
from app.services.omr_utils import detect_bubbles, get_answers, preprocess_and_align


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a BGR numpy array for OpenCV."""
    rgb = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def decode_uploaded_file(raw_bytes: bytes, content_type: str | None) -> np.ndarray:
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    if content_type == "application/pdf":
        pages = convert_from_bytes(raw_bytes, dpi=300)
        if not pages:
            raise ValueError("PDF contains no pages.")
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

    # Try OpenCV first (fast path for JPEG, PNG, etc.)
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    # Fallback to Pillow for formats OpenCV can't handle (HEIF, HEIC, WEBP, TIFF, BMP, …)
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()  # force decode
        return _pil_to_bgr(pil_img)
    except Exception as exc:
        raise ValueError(
            "Unsupported image file or decode failed. "
            "Supported formats: JPEG, PNG, HEIF/HEIC, WEBP, TIFF, BMP, PDF."
        ) from exc


def process_omr(image: np.ndarray, total_questions: int = 100) -> tuple[list[str], int]:
    aligned = preprocess_and_align(image)
    questions, thresholded, bubble_count = detect_bubbles(aligned)
    answers = get_answers(questions, thresholded)
    cleaned = answers[:total_questions]
    if len(cleaned) < total_questions:
        cleaned.extend(["EMPTY"] * (total_questions - len(cleaned)))
    return cleaned, bubble_count


def process_answer_key(
    image: np.ndarray, total_questions: int = 100
) -> tuple[list[str], dict, str]:
    answers, _, _, analysis, analysis_log = extract_answer_key_with_diagnostics(
        image, total_questions=total_questions, ocr_backend="rapidocr"
    )
    cleaned = answers[:total_questions]
    if len(cleaned) < total_questions:
        cleaned.extend(["EMPTY"] * (total_questions - len(cleaned)))
    analysis["meta"]["cleaned_answers"] = int(sum(1 for value in cleaned if value != "EMPTY"))
    return cleaned, analysis, analysis_log


def evaluate_answers(
    student_answers: list[str], correct_answers: list[str], total_questions: int = 100
) -> dict:
    score = 0
    rows: list[dict] = []

    for idx in range(total_questions):
        student = student_answers[idx] if idx < len(student_answers) else "EMPTY"
        correct = correct_answers[idx] if idx < len(correct_answers) else "EMPTY"
        match = student == correct and student != "EMPTY" and correct != "EMPTY"
        if match:
            score += 1

        if correct == "EMPTY":
            status = "MISSING_KEY"
        elif student == "EMPTY":
            status = "NOT_MARKED"
        elif match:
            status = "CORRECT"
        else:
            status = "WRONG"

        rows.append(
            {
                "question": idx + 1,
                "selected": student,
                "correct": correct,
                "status": status,
            }
        )

    correct_count = sum(1 for row in rows if row["status"] == "CORRECT")
    wrong_count = sum(1 for row in rows if row["status"] == "WRONG")
    not_marked_count = sum(1 for row in rows if row["status"] == "NOT_MARKED")
    missing_key_count = sum(1 for row in rows if row["status"] == "MISSING_KEY")

    return {
        "score": score,
        "total_questions": total_questions,
        "stats": {
            "correct": correct_count,
            "wrong": wrong_count,
            "not_marked": not_marked_count,
            "missing_key": missing_key_count,
        },
        "results": rows,
    }
