import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

try:
    from rapidocr import RapidOCR
except ModuleNotFoundError:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ModuleNotFoundError:
        RapidOCR = None

VALID_OPTIONS = {"A", "B", "C", "D"}
OPTION_MISREADS = {
    "0": "D",
    "O": "D",
    "Q": "D",
    "8": "B",
    "4": "A",
}

_RAPIDOCR_ENGINE = None
_OCR_POOL: ProcessPoolExecutor | None = None
_OCR_POOL_WORKERS = 0


def _phase_row(name: str, seconds: float, total_seconds: float) -> dict:
    ms = round(max(0.0, seconds) * 1000.0, 2)
    pct = round((seconds / total_seconds) * 100.0, 2) if total_seconds > 0 else 0.0
    return {"name": name, "ms": ms, "pct": pct}


def _build_ocr_analysis_log(analysis: dict) -> str:
    lines = [
        "OCR Answer-Key Performance Analysis",
        "===================================",
        f"Total time: {analysis['total_ms']} ms",
        "",
        "Phase Breakdown:",
    ]

    for phase in analysis.get("phases", []):
        lines.append(f"- {phase['name']}: {phase['ms']} ms ({phase['pct']}%)")

    meta = analysis.get("meta", {})
    lines.extend(
        [
            "",
            "Run Metadata:",
            f"- total_questions: {meta.get('total_questions', 0)}",
            f"- grid_detected_answers: {meta.get('grid_detected_answers', 0)}",
            f"- fullimage_detected_answers: {meta.get('fullimage_detected_answers', 0)}",
            f"- final_detected_answers: {meta.get('final_detected_answers', 0)}",
            f"- grid_target_questions: {meta.get('grid_target_questions', 0)}",
            f"- ocr_workers: {meta.get('ocr_workers', 1)}",
            f"- ocr_backend: {meta.get('ocr_backend', 'rapidocr')}",
        ]
    )
    return "\n".join(lines)


def _resolve_ocr_backend(ocr_backend: str | None = None) -> str:
    return "rapidocr"


def _get_rapidocr_engine():
    global _RAPIDOCR_ENGINE
    if RapidOCR is None:
        return None
    if _RAPIDOCR_ENGINE is None:
        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _get_ocr_mp_workers() -> int:
    raw = os.getenv("OCR_MP_WORKERS", "1").strip()
    try:
        workers = int(raw)
    except ValueError:
        return 1
    if workers < 1:
        return 1
    return min(workers, max(1, os.cpu_count() or 1))


def _get_ocr_pool(workers: int) -> ProcessPoolExecutor | None:
    global _OCR_POOL, _OCR_POOL_WORKERS
    if workers <= 1:
        return None

    if _OCR_POOL is not None and _OCR_POOL_WORKERS == workers:
        return _OCR_POOL

    if _OCR_POOL is not None and _OCR_POOL_WORKERS != workers:
        _OCR_POOL.shutdown(wait=False, cancel_futures=True)
        _OCR_POOL = None
        _OCR_POOL_WORKERS = 0

    _OCR_POOL = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    )
    _OCR_POOL_WORKERS = workers
    return _OCR_POOL


def _ocr_cell_batch_worker(
    payload: tuple[list[tuple[int, np.ndarray]], str]
) -> list[tuple[int, str | None, np.ndarray, np.ndarray]]:
    batch, ocr_backend = payload
    results: list[tuple[int, str | None, np.ndarray, np.ndarray]] = []
    for job_idx, cell in batch:
        option, preview = _read_option_from_cell(cell, ocr_backend=ocr_backend)
        feature = _cell_feature(preview)
        results.append((job_idx, option, preview, feature))
    return results


def _run_cell_ocr_jobs(
    jobs: list[tuple[int, np.ndarray]], ocr_backend: str
) -> list[tuple[int, str | None, np.ndarray, np.ndarray]]:
    if not jobs:
        return []

    workers = _get_ocr_mp_workers()
    if workers <= 1 or len(jobs) < 20:
        return _ocr_cell_batch_worker((jobs, ocr_backend))

    pool = _get_ocr_pool(workers)
    if pool is None:
        return _ocr_cell_batch_worker((jobs, ocr_backend))

    chunk_size = max(8, len(jobs) // (workers * 2))
    chunks = [jobs[i : i + chunk_size] for i in range(0, len(jobs), chunk_size)]

    try:
        futures = [pool.submit(_ocr_cell_batch_worker, (chunk, ocr_backend)) for chunk in chunks]
        merged: list[tuple[int, str | None, np.ndarray, np.ndarray]] = []
        for future in as_completed(futures):
            merged.extend(future.result())
        return merged
    except Exception:
        return _ocr_cell_batch_worker((jobs, ocr_backend))


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = _order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _extract_table_region(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    image_area = image.shape[0] * image.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.2:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        warped = _four_point_transform(image, approx.reshape(4, 2).astype(np.float32))
        if warped.shape[0] > 100 and warped.shape[1] > 100:
            return warped

    return image.copy()


def _prepare_ocr_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = _extract_table_region(image)
    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    ocr_ready = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    return table, ocr_ready


def _cluster_positions(indices: np.ndarray, max_gap: int = 4) -> list[int]:
    if indices.size == 0:
        return []

    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])

    return [int(round(float(np.mean(group)))) for group in groups if group]


def _projection_line_positions(line_mask: np.ndarray, axis: int) -> list[int]:
    projection = np.sum(line_mask > 0, axis=axis).astype(np.float32)
    if projection.size == 0 or float(projection.max(initial=0.0)) <= 0.0:
        return []

    threshold = float(projection.max()) * 0.35
    indices = np.where(projection >= threshold)[0]
    return _cluster_positions(indices, max_gap=6)


def _fit_boundaries(
    detected: list[int], expected_segments: int, image_extent: int
) -> list[int]:
    expected_lines = expected_segments + 1
    if image_extent <= 1:
        return [0] * expected_lines

    if len(detected) >= 2:
        start = max(0, min(image_extent - 1, detected[0]))
        end = max(start + 1, min(image_extent - 1, detected[-1]))
    else:
        start = 0
        end = image_extent - 1

    ideal = np.linspace(start, end, expected_lines)
    if len(detected) >= expected_lines:
        detected_arr = np.array(sorted(detected), dtype=np.float32)
        fitted = []
        for point in ideal:
            nearest = int(detected_arr[np.argmin(np.abs(detected_arr - point))])
            fitted.append(nearest)
    else:
        fitted = [int(round(point)) for point in ideal]

    fitted[0] = 0
    fitted[-1] = image_extent - 1
    for idx in range(1, len(fitted)):
        if fitted[idx] <= fitted[idx - 1]:
            fitted[idx] = min(image_extent - 1, fitted[idx - 1] + 1)

    return fitted


def _detect_table_boundaries(table: np.ndarray) -> tuple[list[int], list[int]]:
    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    h, w = binary_inv.shape
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 22)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 22), 1))

    vertical_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel)

    x_positions = _projection_line_positions(vertical_lines, axis=0)
    y_positions = _projection_line_positions(horizontal_lines, axis=1)

    x_bounds = _fit_boundaries(x_positions, expected_segments=10, image_extent=w)
    y_bounds = _fit_boundaries(y_positions, expected_segments=20, image_extent=h)
    return x_bounds, y_bounds


def _stabilize_bounds(bounds: list[int], expected_segments: int, image_extent: int) -> list[int]:
    fallback = [int(round(i * max(1, image_extent - 1) / expected_segments)) for i in range(expected_segments + 1)]
    if len(bounds) != expected_segments + 1:
        return fallback

    clipped = [max(0, min(image_extent - 1, int(value))) for value in bounds]
    clipped[0] = 0
    clipped[-1] = image_extent - 1
    for idx in range(1, len(clipped)):
        if clipped[idx] <= clipped[idx - 1]:
            clipped[idx] = min(image_extent - 1, clipped[idx - 1] + 1)

    widths = np.diff(np.array(clipped, dtype=np.float32))
    if widths.size == 0:
        return fallback

    median_w = float(np.median(widths))
    if median_w <= 0:
        return fallback

    too_small = widths < max(3.0, median_w * 0.34)
    too_large = widths > (median_w * 2.6)
    if int(np.sum(np.logical_or(too_small, too_large))) > max(2, expected_segments // 4):
        return fallback
    return clipped


def _rapidocr_candidates(image: np.ndarray) -> list[tuple[str, float]]:
    engine = _get_rapidocr_engine()
    if engine is None:
        return []

    input_image = image
    if image.ndim == 2:
        input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    try:
        result, _ = engine(input_image)
    except Exception:
        return []

    if not result:
        return []

    candidates: list[tuple[str, float]] = []
    for item in result:
        if len(item) < 3:
            continue
        text = re.sub(r"[^A-Z0-9]", "", str(item[1]).upper())
        if not text:
            continue
        conf = float(item[2]) * 100.0
        for idx, ch in enumerate(text):
            option = _normalize_option(ch)
            if option is None:
                continue
            position_boost = 1.0 if idx == 0 else 0.72
            candidates.append((option, conf * position_boost))
            break
    return candidates


def _clean_binary_artifacts(binary: np.ndarray) -> np.ndarray:
    cleaned = binary.copy()
    if cleaned.ndim != 2 or cleaned.size == 0:
        return cleaned

    h, w = cleaned.shape
    ink_mask = cleaned < 128
    if not np.any(ink_mask):
        return cleaned

    col_ratio = np.mean(ink_mask, axis=0)
    row_ratio = np.mean(ink_mask, axis=1)

    # Remove near-solid bars, typically leaked table borders after crop/resize.
    bad_cols = np.where(col_ratio >= 0.94)[0]
    bad_rows = np.where(row_ratio >= 0.94)[0]
    if bad_cols.size:
        cleaned[:, bad_cols] = 255
    if bad_rows.size:
        cleaned[bad_rows, :] = 255

    # Remove heavy edge bars from left/right/top/bottom bands.
    edge_w = max(1, int(w * 0.12))
    edge_h = max(1, int(h * 0.12))
    left_cols = np.where(col_ratio[:edge_w] >= 0.62)[0]
    right_cols = np.where(col_ratio[-edge_w:] >= 0.62)[0] + (w - edge_w)
    top_rows = np.where(row_ratio[:edge_h] >= 0.62)[0]
    bottom_rows = np.where(row_ratio[-edge_h:] >= 0.62)[0] + (h - edge_h)
    if left_cols.size:
        cleaned[:, left_cols] = 255
    if right_cols.size:
        cleaned[:, right_cols] = 255
    if top_rows.size:
        cleaned[top_rows, :] = 255
    if bottom_rows.size:
        cleaned[bottom_rows, :] = 255

    # Remove very large border-connected components (line chunks, corner fills).
    inv = (cleaned < 128).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        if area <= 0:
            continue
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if not touches_border:
            continue
        area_ratio = float(area) / float(h * w)
        tall_bar = bh >= int(h * 0.78) and bw <= int(max(2, w * 0.22))
        wide_bar = bw >= int(w * 0.78) and bh <= int(max(2, h * 0.22))
        if area_ratio >= 0.08 or tall_bar or wide_bar:
            cleaned[labels == idx] = 255

    return cleaned


def _select_letter_components(binary_inv: np.ndarray) -> np.ndarray:
    if binary_inv.ndim != 2 or binary_inv.size == 0:
        return binary_inv

    h, w = binary_inv.shape
    if h < 4 or w < 4:
        return binary_inv

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    if num <= 1:
        return binary_inv

    image_area = float(h * w)
    min_area = max(10.0, image_area * 0.001)
    cx0 = (w - 1) / 2.0
    cy0 = (h - 1) / 2.0
    max_dist = float(np.hypot(cx0, cy0)) + 1e-6

    candidates: list[tuple[int, float]] = []
    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        if area < min_area:
            continue

        # Reject bar-like connected components from table borders.
        if bw <= max(2, int(w * 0.10)) and bh >= int(h * 0.70):
            continue
        if bh <= max(2, int(h * 0.10)) and bw >= int(w * 0.70):
            continue

        touches_edge = x <= 0 or y <= 0 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1)
        comp_cx, comp_cy = centroids[idx]
        dist = float(np.hypot(comp_cx - cx0, comp_cy - cy0)) / max_dist
        fill_ratio = float(area) / float(max(1, bw * bh))
        aspect = float(bw) / float(max(1, bh))

        shape_bonus = 1.0
        if 0.16 <= aspect <= 1.9:
            shape_bonus += 0.25
        if fill_ratio <= 0.86:
            shape_bonus += 0.15

        edge_penalty = 0.55 if touches_edge else 1.0
        score = float(area) * shape_bonus * edge_penalty * (1.2 - min(1.0, dist))
        if score <= 0:
            continue
        candidates.append((idx, score))

    if not candidates:
        return binary_inv

    candidates.sort(key=lambda item: item[1], reverse=True)
    top_score = candidates[0][1]
    keep_ids = [idx for idx, score in candidates if score >= (top_score * 0.45)]
    keep_ids = keep_ids[:3]
    if not keep_ids:
        keep_ids = [candidates[0][0]]

    filtered = np.zeros_like(binary_inv)
    for idx in keep_ids:
        filtered[labels == idx] = 255
    return filtered


def _read_option_from_cell(cell: np.ndarray, ocr_backend: str = "auto") -> tuple[str | None, np.ndarray]:
    if cell.size == 0:
        return None, np.full((8, 8), 255, dtype=np.uint8)

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    preview_seed = cv2.copyMakeBorder(gray, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    preview_seed = cv2.resize(preview_seed, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    preview_seed = cv2.threshold(preview_seed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    best_preview = _clean_binary_artifacts(preview_seed)

    scores: dict[str, float] = {}
    best_preview_score = -1.0
    crop_profiles = [(0.18, 0.16), (0.24, 0.22)]

    for crop_idx, (trim_x_ratio, trim_y_ratio) in enumerate(crop_profiles):
        trim_x = max(1, int(w * trim_x_ratio))
        trim_y = max(1, int(h * trim_y_ratio))
        x0 = min(trim_x, w - 1)
        x1 = max(x0 + 1, w - trim_x)
        y0 = min(trim_y, h - 1)
        y1 = max(y0 + 1, h - trim_y)
        core = gray[y0:y1, x0:x1]
        if core.size == 0:
            continue

        core = cv2.copyMakeBorder(core, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
        scaled = cv2.resize(core, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.GaussianBlur(scaled, (3, 3), 0)

        bin_otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        bin_otsu = cv2.bitwise_not(bin_otsu_inv)
        h2, w2 = bin_otsu_inv.shape

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, h2 // 5)))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, w2 // 5), 1))
        v_lines = cv2.morphologyEx(bin_otsu_inv, cv2.MORPH_OPEN, vertical_kernel)
        h_lines = cv2.morphologyEx(bin_otsu_inv, cv2.MORPH_OPEN, horizontal_kernel)
        cleaned_inv = cv2.subtract(bin_otsu_inv, cv2.bitwise_or(v_lines, h_lines))
        cleaned_inv = cv2.medianBlur(cleaned_inv, 3)
        cleaned_inv = _select_letter_components(cleaned_inv)
        cleaned = cv2.bitwise_not(cleaned_inv)

        coords = cv2.findNonZero(cleaned_inv)
        focused = cleaned
        if coords is not None:
            gx, gy, gw, gh = cv2.boundingRect(coords)
            mask_area = float(gw * gh)
            full_area = float(cleaned_inv.shape[0] * cleaned_inv.shape[1])
            if full_area > 0 and 0.01 <= (mask_area / full_area) <= 0.92:
                pad = max(4, int(min(cleaned_inv.shape[:2]) * 0.06))
                fx0 = max(0, gx - pad)
                fy0 = max(0, gy - pad)
                fx1 = min(cleaned_inv.shape[1], gx + gw + pad)
                fy1 = min(cleaned_inv.shape[0], gy + gh + pad)
                focused_inv = cleaned_inv[fy0:fy1, fx0:fx1]
                focused_inv = cv2.copyMakeBorder(
                    focused_inv, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0
                )
                focused = cv2.bitwise_not(focused_inv)

        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            9,
        )
        adaptive = cv2.bitwise_not(adaptive)
        bin_otsu = _clean_binary_artifacts(bin_otsu)
        cleaned = _clean_binary_artifacts(cleaned)
        focused = _clean_binary_artifacts(focused)
        adaptive = _clean_binary_artifacts(adaptive)
        variants: list[tuple[np.ndarray, float]] = [
            (focused, 1.10),
            (cleaned, 1.00),
            (bin_otsu, 0.95),
            (adaptive, 0.90),
        ]

        for variant, variant_weight in variants:
            ink_ratio = float(np.mean(variant < 128))
            if ink_ratio < 0.0015 or ink_ratio > 0.62:
                continue

            variant_total = 0.0
            for option, conf in _rapidocr_candidates(variant):
                score = max(0.0, conf) * variant_weight
                scores[option] = scores.get(option, 0.0) + score
                variant_total += score

            if variant_total > best_preview_score:
                best_preview_score = variant_total
                best_preview = variant

            if scores and max(scores.values()) >= 98.0:
                break
            if scores and max(scores.values()) >= 82.0:
                break

        if scores:
            top_score = max(scores.values())
            if top_score >= 98.0:
                break
            if crop_idx == 0 and top_score >= 82.0:
                break

    if not scores:
        fallback_gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
        fallback_gray = cv2.resize(fallback_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        fallback_bin = cv2.threshold(
            fallback_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        fallback_bin = _clean_binary_artifacts(fallback_bin)
        for option, conf in _rapidocr_candidates(fallback_gray):
            score = max(0.0, conf) * 0.85
            scores[option] = scores.get(option, 0.0) + score
            if score > best_preview_score:
                best_preview_score = score
            best_preview = fallback_bin

    if not scores:
        return None, best_preview

    best_option = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return best_option, best_preview


def _cell_feature(binary_cell: np.ndarray) -> np.ndarray:
    resized = cv2.resize(binary_cell, (28, 28), interpolation=cv2.INTER_AREA)
    ink = (resized < 128).astype(np.float32)
    total = float(np.sum(ink))
    if total > 0:
        ink = ink / total
    return ink.flatten()


def _extract_answers_by_grid(
    table: np.ndarray,
    total_questions: int,
    ocr_backend: str = "auto",
    target_questions: set[int] | None = None,
) -> tuple[list[str], np.ndarray]:
    answers = ["EMPTY"] * total_questions
    debug_view = np.full(table.shape[:2], 255, dtype=np.uint8)
    features_by_option: dict[str, list[np.ndarray]] = {option: [] for option in VALID_OPTIONS}
    empty_cells: list[tuple[int, np.ndarray]] = []
    cell_jobs: list[tuple[int, np.ndarray]] = []
    cell_meta: dict[int, tuple[int, int, int, int, int]] = {}

    h, w = table.shape[:2]
    detected_x, detected_y = _detect_table_boundaries(table)
    x_bounds = _stabilize_bounds(detected_x, expected_segments=10, image_extent=w)
    y_bounds = _stabilize_bounds(detected_y, expected_segments=20, image_extent=h)
    x_steps = np.diff(np.array(x_bounds, dtype=np.int32))
    y_steps = np.diff(np.array(y_bounds, dtype=np.int32))
    if x_steps.size != 10 or int(np.sum(x_steps < 8)) > 2:
        x_bounds = [int(round(i * w / 10.0)) for i in range(11)]
    if y_steps.size != 20 or int(np.sum(y_steps < 6)) > 3:
        y_bounds = [int(round(i * h / 20.0)) for i in range(21)]
    option_columns = [1, 3, 5, 7, 9]

    for row_idx in range(min(20, len(y_bounds) - 1)):
        y0, y1 = y_bounds[row_idx], y_bounds[row_idx + 1]
        if y1 - y0 < 6:
            continue

        for group_idx, col_idx in enumerate(option_columns):
            if col_idx + 1 >= len(x_bounds):
                continue

            x0, x1 = x_bounds[col_idx], x_bounds[col_idx + 1]
            if x1 - x0 < 6:
                continue

            pad_x = max(1, int((x1 - x0) * 0.05))
            pad_y = max(1, int((y1 - y0) * 0.08))

            cx0 = min(max(x0 + pad_x, x0), x1 - 1)
            cx1 = max(min(x1 - pad_x, x1), cx0 + 1)
            cy0 = min(max(y0 + pad_y, y0), y1 - 1)
            cy1 = max(min(y1 - pad_y, y1), cy0 + 1)
            cell = table[cy0:cy1, cx0:cx1]
            question_number = row_idx + 1 + (group_idx * 20)
            if target_questions is not None and question_number not in target_questions:
                continue

            job_idx = len(cell_jobs)
            cell_jobs.append((job_idx, cell.copy()))
            cell_meta[job_idx] = (question_number, y0, y1, x0, x1)

    for job_idx, option, preview, feature in _run_cell_ocr_jobs(cell_jobs, ocr_backend=ocr_backend):
        question_number, y0, y1, x0, x1 = cell_meta[job_idx]
        if 1 <= question_number <= total_questions:
            if option is not None:
                answers[question_number - 1] = option
                features_by_option[option].append(feature)
            else:
                empty_cells.append((question_number - 1, feature))

        preview_resized = cv2.resize(preview, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        debug_view[y0:y1, x0:x1] = preview_resized

    prototypes: dict[str, np.ndarray] = {}
    for option, vectors in features_by_option.items():
        if len(vectors) >= 4:
            proto = np.mean(np.stack(vectors, axis=0), axis=0)
            norm = float(np.linalg.norm(proto))
            if norm > 0:
                prototypes[option] = proto / norm

    for idx, feature in empty_cells:
        if answers[idx] != "EMPTY" or not prototypes:
            continue
        fnorm = float(np.linalg.norm(feature))
        if fnorm <= 0:
            continue
        feature_norm = feature / fnorm
        scored = sorted(
            ((option, float(np.dot(feature_norm, proto))) for option, proto in prototypes.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        best_option, best_score = scored[0]
        second_best = scored[1][1] if len(scored) > 1 else -1.0
        if best_score >= 0.56 and (best_score - second_best) >= 0.02:
            answers[idx] = best_option

    # Second pass: recover remaining blanks with softer threshold when prototype bank is strong.
    if len(prototypes) >= 3:
        for idx, feature in empty_cells:
            if answers[idx] != "EMPTY":
                continue
            fnorm = float(np.linalg.norm(feature))
            if fnorm <= 0:
                continue
            feature_norm = feature / fnorm
            scored = sorted(
                ((option, float(np.dot(feature_norm, proto))) for option, proto in prototypes.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            best_option, best_score = scored[0]
            second_best = scored[1][1] if len(scored) > 1 else -1.0
            if best_score >= 0.48 and (best_score - second_best) >= 0.008:
                answers[idx] = best_option

    # Third pass: if we already recognized a strong majority, fill hard leftovers conservatively.
    recognized = sum(1 for value in answers if value != "EMPTY")
    if recognized >= int(total_questions * 0.72) and len(prototypes) >= 3:
        for idx, feature in empty_cells:
            if answers[idx] != "EMPTY":
                continue
            fnorm = float(np.linalg.norm(feature))
            if fnorm <= 0:
                continue
            feature_norm = feature / fnorm
            scored = sorted(
                ((option, float(np.dot(feature_norm, proto))) for option, proto in prototypes.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            best_option, best_score = scored[0]
            second_best = scored[1][1] if len(scored) > 1 else -1.0
            if best_score >= 0.40 and (best_score - second_best) >= 0.0:
                answers[idx] = best_option

    return answers, debug_view


def _normalize_option(text: str) -> str | None:
    token = text.strip().upper()
    if token in VALID_OPTIONS:
        return token
    if token in OPTION_MISREADS:
        return OPTION_MISREADS[token]
    return None


def _extract_pairs_from_tokens(
    tokens: list[dict], total_questions: int, image_width: int
) -> dict[int, str]:
    number_tokens = []
    option_tokens = []

    for token in tokens:
        text = token["text"]
        if text.isdigit():
            value = int(text)
            if 1 <= value <= total_questions:
                number_tokens.append({**token, "value": value})
            continue

        option = _normalize_option(text)
        if option is not None:
            option_tokens.append({**token, "value": option, "used": False})

    if not number_tokens or not option_tokens:
        return {}

    median_h = float(np.median([token["h"] for token in tokens])) if tokens else 20.0
    row_tolerance = max(10.0, median_h * 1.1)
    max_pair_distance = max(40.0, image_width * 0.18)

    extracted: dict[int, tuple[str, float]] = {}
    for number in sorted(number_tokens, key=lambda t: (t["cy"], t["cx"])):
        best_option = None
        best_score = float("inf")

        for option in option_tokens:
            if option["used"]:
                continue

            dx = option["cx"] - number["cx"]
            if dx <= 0 or dx > max_pair_distance:
                continue

            dy = abs(option["cy"] - number["cy"])
            if dy > row_tolerance:
                continue

            score = dx + (dy * 3.0)
            if score < best_score:
                best_score = score
                best_option = option

        if best_option is None:
            continue

        best_option["used"] = True
        confidence = float(number["conf"] + best_option["conf"])
        q_no = int(number["value"])
        option_val = str(best_option["value"])

        if q_no not in extracted or confidence > extracted[q_no][1]:
            extracted[q_no] = (option_val, confidence)

    return {q_no: pair[0] for q_no, pair in extracted.items()}


def _extract_pairs_from_text(text: str, total_questions: int) -> dict[int, str]:
    fallback: dict[int, str] = {}
    for q_no_text, option_text in re.findall(r"(\d{1,3})\s*([ABCDOQ04])", text.upper()):
        q_no = int(q_no_text)
        if not (1 <= q_no <= total_questions):
            continue
        option = _normalize_option(option_text)
        if option is None:
            continue
        fallback[q_no] = option
    return fallback


def _fullimage_rapidocr_pass(
    table: np.ndarray, total_questions: int
) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    """Run RapidOCR on the full table image and spatially pair numbers with options."""
    engine = _get_rapidocr_engine()
    if engine is None:
        return {}, {}

    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    variants: list[tuple[str, np.ndarray, float, float]] = [
        ("table", table, 1.0, 1.0),
        ("enhanced", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 1.0, 1.0),
    ]
    # Add a sharpened variant
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
    variants.append(("sharpened", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), 1.0, 1.0))

    # Add upscaled variant for small text
    upscaled = cv2.resize(enhanced, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscaled", cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR), 1.0 / 1.5, 2.6))

    # Add adaptive threshold variant for screenshot/PDF artifacts
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        15,
    )
    variants.append(("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR), 1.0, 0.9))

    all_answers: list[tuple[dict[int, str], float]] = []

    for _, img_variant, coord_scale, vote_weight in variants:
        try:
            result, _ = engine(img_variant)
        except Exception:
            continue
        if not result:
            continue

        tokens: list[dict] = []
        for bbox, text, conf in result:
            cx = sum(p[0] for p in bbox) / 4.0 * coord_scale
            cy = sum(p[1] for p in bbox) / 4.0 * coord_scale
            bw = (max(p[0] for p in bbox) - min(p[0] for p in bbox)) * coord_scale
            bh = (max(p[1] for p in bbox) - min(p[1] for p in bbox)) * coord_scale
            tokens.append({
                "text": text.strip().upper(),
                "conf": float(conf),
                "cx": cx, "cy": cy,
                "w": bw, "h": bh,
            })

        if not tokens:
            continue

        # Try to extract paired answers from spatial layout
        pairs = _extract_pairs_from_tokens(tokens, total_questions,
                                           int(table.shape[1]))
        if pairs:
            all_answers.append((pairs, vote_weight))

    if not all_answers:
        return {}, {}

    # Majority vote across variants
    merged: dict[int, str] = {}
    vote_meta: dict[int, tuple[int, int, int]] = {}
    for q in range(1, total_questions + 1):
        votes: dict[str, int] = {}
        weighted_votes: dict[str, float] = {}
        for ans_dict, vote_weight in all_answers:
            v = ans_dict.get(q)
            if v and v != "EMPTY":
                votes[v] = votes.get(v, 0) + 1
                weighted_votes[v] = weighted_votes.get(v, 0.0) + float(vote_weight)
        if votes:
            sorted_votes = sorted(
                votes.items(),
                key=lambda item: (
                    -weighted_votes.get(item[0], 0.0),
                    -item[1],
                    item[0],
                ),
            )
            best_answer, best_count = sorted_votes[0]
            second_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
            total_count = int(sum(votes.values()))
            merged[q] = best_answer
            vote_meta[q] = (int(best_count), total_count, int(best_count - second_count))
    return merged, vote_meta


def extract_answer_key_with_diagnostics(
    image: np.ndarray, total_questions: int = 100, ocr_backend: str = "rapidocr"
) -> tuple[list[str], np.ndarray, np.ndarray, dict, str]:
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    phase_start = time.perf_counter()
    has_rapidocr = _get_rapidocr_engine() is not None
    if not has_rapidocr:
        raise RuntimeError("RapidOCR backend not available. Install rapidocr-onnxruntime.")

    if image is None or image.size == 0:
        raise ValueError("Answer key image is empty or invalid.")
    phase_seconds["validation_and_engine_check"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    table_view, ocr_ready = _prepare_ocr_image(image)
    phase_seconds["image_preparation"] = time.perf_counter() - phase_start

    # Pass 1: full-image RapidOCR with spatial pairing
    phase_start = time.perf_counter()
    fullimg_answers, fullimg_vote_meta = _fullimage_rapidocr_pass(table_view, total_questions)
    phase_seconds["fullimage_ocr_pass"] = time.perf_counter() - phase_start

    # Decide where expensive grid OCR is still needed.
    # Criteria: full-image missing answer OR weak consensus.
    low_confidence_questions: set[int] = set()
    for q in range(1, total_questions + 1):
        if q not in fullimg_answers:
            low_confidence_questions.add(q)
            continue
        best_count, total_count, margin = fullimg_vote_meta.get(q, (0, 0, 0))
        has_strong_consensus = best_count >= 2 and total_count >= 2 and margin >= 1
        if not has_strong_consensus:
            low_confidence_questions.add(q)

    grid_target_questions = sorted(low_confidence_questions)

    # Pass 2: grid OCR only for targeted questions (can be empty if full-image is strong).
    phase_start = time.perf_counter()
    if grid_target_questions:
        grid_answers, grid_debug_view = _extract_answers_by_grid(
            table_view,
            total_questions,
            ocr_backend="rapidocr",
            target_questions=set(grid_target_questions),
        )
    else:
        grid_answers = ["EMPTY"] * total_questions
        grid_debug_view = np.full(table_view.shape[:2], 255, dtype=np.uint8)
    phase_seconds["grid_cell_ocr"] = time.perf_counter() - phase_start

    # Merge strategy:
    # - Start with full-image answers (fast path)
    # - Use grid only for unresolved or low-confidence full-image items
    phase_start = time.perf_counter()
    final_answers = ["EMPTY"] * total_questions
    for q in range(1, total_questions + 1):
        idx = q - 1
        full_val = fullimg_answers.get(q)
        grid_val = grid_answers[idx]

        if full_val is not None:
            final_answers[idx] = full_val

        if q in low_confidence_questions and grid_val != "EMPTY":
            final_answers[idx] = grid_val

        if final_answers[idx] == "EMPTY" and grid_val != "EMPTY":
            final_answers[idx] = grid_val
    phase_seconds["merge_and_consensus"] = time.perf_counter() - phase_start

    total_seconds = time.perf_counter() - started
    phases = [
        _phase_row("validation_and_engine_check", phase_seconds.get("validation_and_engine_check", 0.0), total_seconds),
        _phase_row("image_preparation", phase_seconds.get("image_preparation", 0.0), total_seconds),
        _phase_row("grid_cell_ocr", phase_seconds.get("grid_cell_ocr", 0.0), total_seconds),
        _phase_row("fullimage_ocr_pass", phase_seconds.get("fullimage_ocr_pass", 0.0), total_seconds),
        _phase_row("merge_and_consensus", phase_seconds.get("merge_and_consensus", 0.0), total_seconds),
    ]
    analysis = {
        "total_ms": round(total_seconds * 1000.0, 2),
        "phases": phases,
        "meta": {
            "total_questions": total_questions,
            "grid_detected_answers": int(sum(1 for value in grid_answers if value != "EMPTY")),
            "fullimage_detected_answers": int(len(fullimg_answers)),
            "final_detected_answers": int(sum(1 for value in final_answers if value != "EMPTY")),
            "grid_target_questions": int(len(grid_target_questions)),
            "ocr_workers": _get_ocr_mp_workers(),
            "ocr_backend": ocr_backend,
        },
    }
    analysis_log = _build_ocr_analysis_log(analysis)

    return final_answers, table_view, grid_debug_view, analysis, analysis_log


def extract_answer_key(
    image: np.ndarray, total_questions: int = 100, ocr_backend: str = "rapidocr"
) -> tuple[list[str], np.ndarray, np.ndarray]:
    answers, table_view, grid_debug_view, _, _ = extract_answer_key_with_diagnostics(
        image, total_questions=total_questions, ocr_backend=ocr_backend
    )
    return answers, table_view, grid_debug_view
