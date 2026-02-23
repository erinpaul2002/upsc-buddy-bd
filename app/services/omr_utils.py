import cv2
import imutils
import numpy as np


def _kmeans_1d(values: list[float], clusters: int) -> tuple[np.ndarray, np.ndarray] | None:
    if len(values) < clusters:
        return None

    samples = np.float32(values).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    _, labels, centers = cv2.kmeans(
        samples, clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    centers = centers.flatten()
    order = np.argsort(centers)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(clusters)

    ranked_labels = np.array([ranks[label[0]] for label in labels], dtype=np.int32)
    sorted_centers = centers[order]
    return ranked_labels, sorted_centers


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def preprocess_and_align(
    image: np.ndarray, canny_low: int = 75, canny_high: int = 200
) -> np.ndarray:
    ratio = image.shape[0] / 500.0
    original = image.copy()
    resized = imutils.resize(image, height=500)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            screen_contour = approx
            break

    if screen_contour is None:
        fallback_h = int(1000 * original.shape[0] / original.shape[1])
        return cv2.resize(original, (1000, fallback_h))

    warped = four_point_transform(original, screen_contour.reshape(4, 2) * ratio)
    warped_h = int(1000 * warped.shape[0] / warped.shape[1])
    return cv2.resize(warped, (1000, warped_h))


def detect_bubbles(
    warped: np.ndarray,
    min_area: int = 80,
    max_area: int = 600,
    roi_top_ratio: float = 0.43,
    roi_bottom_ratio: float = 0.92,
    roi_left_ratio: float = 0.09,
    roi_right_ratio: float = 0.93,
) -> tuple[list[list[dict]], np.ndarray, int]:
    sheet_h, sheet_w = warped.shape[:2]
    y0 = int(max(0.0, min(0.95, roi_top_ratio)) * sheet_h)
    y1 = int(max(0.05, min(1.0, roi_bottom_ratio)) * sheet_h)
    x0 = int(max(0.0, min(0.95, roi_left_ratio)) * sheet_w)
    x1 = int(max(0.05, min(1.0, roi_right_ratio)) * sheet_w)

    if y1 <= y0 or x1 <= x0:
        gray_full = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresholded_full = cv2.threshold(
            gray_full, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        return [], thresholded_full, 0

    answer_roi = warped[y0:y1, x0:x1]
    gray_roi = cv2.cvtColor(answer_roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    thresholded_roi = cv2.adaptiveThreshold(
        blurred_roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    contours = cv2.findContours(
        thresholded_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        if not (0.8 <= aspect <= 1.3):
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * area) / max(perimeter * perimeter, 1e-6)
        if not (0.55 <= circularity <= 1.35):
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        local_cx = int(moments["m10"] / moments["m00"])
        local_cy = int(moments["m01"] / moments["m00"])
        global_contour = contour + np.array([[[x0, y0]]], dtype=np.int32)
        bubbles.append(
            {
                "cnt": global_contour,
                "cx": local_cx + x0,
                "cy": local_cy + y0,
                "area": area,
                "local_cx": local_cx,
                "local_cy": local_cy,
            }
        )

    bubble_count = len(bubbles)
    gray_full = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresholded_full = cv2.threshold(
        gray_full, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    if bubble_count < 80:
        return [], thresholded_full, bubble_count

    row_cluster = _kmeans_1d([bubble["cy"] for bubble in bubbles], 20)
    col_cluster = _kmeans_1d([bubble["cx"] for bubble in bubbles], 5)
    if row_cluster is None or col_cluster is None:
        return [], thresholded_full, bubble_count

    row_labels, _ = row_cluster
    col_labels, _ = col_cluster
    for index, bubble in enumerate(bubbles):
        bubble["row_idx"] = int(row_labels[index])
        bubble["col_idx"] = int(col_labels[index])

    option_centers: dict[int, np.ndarray] = {}
    for col_idx in range(5):
        col_bubbles = [bubble for bubble in bubbles if bubble["col_idx"] == col_idx]
        option_cluster = _kmeans_1d([bubble["cx"] for bubble in col_bubbles], 4)
        if option_cluster is None:
            continue
        option_centers[col_idx] = option_cluster[1]

    questions: list[list[dict | None]] = [[None, None, None, None] for _ in range(100)]
    option_distances: list[list[float]] = [[1e9, 1e9, 1e9, 1e9] for _ in range(100)]

    for bubble in bubbles:
        row_idx = bubble["row_idx"]
        col_idx = bubble["col_idx"]
        if col_idx not in option_centers:
            continue

        q_idx = (col_idx * 20) + row_idx
        if q_idx < 0 or q_idx >= 100:
            continue

        centers = option_centers[col_idx]
        option_idx = int(np.argmin(np.abs(centers - bubble["cx"])))
        distance = float(abs(centers[option_idx] - bubble["cx"]))

        if distance < option_distances[q_idx][option_idx]:
            option_distances[q_idx][option_idx] = distance
            questions[q_idx][option_idx] = bubble

    return questions, thresholded_full, bubble_count


def get_answers(
    questions: list[list[dict | None]], thresholded: np.ndarray, fill_threshold: float = 0.45
) -> list[str]:
    answers = []
    for row in questions[:100]:
        selected = "EMPTY"
        max_fill = 0.0

        for index, bubble in enumerate(row):
            if bubble is None:
                continue

            mask = np.zeros(thresholded.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble["cnt"]], -1, 255, -1)
            mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            mask_pixels = cv2.countNonZero(mask)
            if mask_pixels == 0:
                continue

            filled = cv2.countNonZero(cv2.bitwise_and(thresholded, mask))
            fill_ratio = filled / float(mask_pixels)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                selected = chr(65 + index)

        answers.append(selected if max_fill > fill_threshold else "EMPTY")

    if len(answers) < 100:
        answers.extend(["EMPTY"] * (100 - len(answers)))

    return answers[:100]
