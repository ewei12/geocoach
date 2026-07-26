# road_markings.py
"""
Detects and interprets road/lane markings for GeoGuessr-style reasoning:
line color (white/yellow), pattern (solid/dashed), count (single/double).
"""

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _color_masks(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    white_mask = ((s < 60) & (v > 170)).astype(np.uint8) * 255
    yellow_mask = ((h >= 20) & (h <= 32) & (s > 130) & (v > 120)).astype(np.uint8) * 255

    return white_mask, yellow_mask


def _clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _line_segments_from_mask(mask):
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180,
        threshold=40, minLineLength=20, maxLineGap=8,
    )
    if lines is None:
        return []
    return [tuple(np.asarray(line).reshape(-1)[:4]) for line in lines]


def _segment_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def _group_by_angle_and_offset(segments, angle_tol=12, offset_bins=40):
    if not segments:
        return []

    groups = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        angle = _segment_angle(x1, y1, x2, y2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        placed = False
        for g in groups:
            angle_diff = min(abs(angle - g["angle"]), 180 - abs(angle - g["angle"]))
            nearest_dist = min(
                np.hypot(mid_x - (s[0] + s[2]) / 2, mid_y - (s[1] + s[3]) / 2)
                for s in g["segs"]
            )
            if angle_diff < angle_tol and nearest_dist < offset_bins:
                g["segs"].append(seg)
                g["angle"] = np.mean([_segment_angle(*s) for s in g["segs"]])
                placed = True
                break

        if not placed:
            groups.append({"angle": angle, "segs": [seg]})

    return [g["segs"] for g in groups if len(g["segs"]) >= 2]


def _classify_pattern(segs):
    if not segs:
        return "unknown", 0.0

    xs = [p for seg in segs for p in (seg[0], seg[2])]
    ys = [p for seg in segs for p in (seg[1], seg[3])]
    span = max(np.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1)
    covered = sum(np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segs)
    fill_ratio = covered / span

    if fill_ratio > 0.75 and len(segs) <= 3:
        return "solid", fill_ratio
    return "dashed", fill_ratio

def _fit_line_through_group(segs):
    pts = np.array(
        [[(s[0] + s[2]) / 2, (s[1] + s[3]) / 2] for s in segs], dtype=np.float32
    )
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0]), np.array([-vy, vx])  # point on line, unit normal


def _perp_distance_between_groups(segs_a, segs_b):
    point_a, normal_a = _fit_line_through_group(segs_a)
    centroid_b = np.mean(
        [[(s[0] + s[2]) / 2, (s[1] + s[3]) / 2] for s in segs_b], axis=0
    )
    return abs(np.dot(centroid_b - point_a, normal_a))


def _y_ranges_overlap(segs_a, segs_b, min_overlap_frac=0.3):
    ys_a = [p for seg in segs_a for p in (seg[1], seg[3])]
    ys_b = [p for seg in segs_b for p in (seg[1], seg[3])]
    lo, hi = max(min(ys_a), min(ys_b)), min(max(ys_a), max(ys_b))
    overlap = max(0, hi - lo)
    span = min(max(ys_a) - min(ys_a) or 1, max(ys_b) - min(ys_b) or 1)
    return overlap / span >= min_overlap_frac

def _detect_doubles(groups, img_width, offset_range_frac=(0.006, 0.025)):
    if len(groups) < 2:
        return False

    parallel_offset_range = (
        offset_range_frac[0] * img_width,
        offset_range_frac[1] * img_width,
    )

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if not _y_ranges_overlap(groups[i], groups[j]):
                continue
            perp_dist = _perp_distance_between_groups(groups[i], groups[j])
            if parallel_offset_range[0] <= perp_dist <= parallel_offset_range[1]:
                return True
    return False

def _stripe_geometry(mask, segs, max_width_frac=0.008, min_aspect=6.0):
    xs = [p for seg in segs for p in (seg[0], seg[2])]
    ys = [p for seg in segs for p in (seg[1], seg[3])]
    x0, x1 = max(0, min(xs) - 15), min(mask.shape[1], max(xs) + 15)
    y0, y1 = max(0, min(ys) - 15), min(mask.shape[0], max(ys) + 15)

    local = mask[int(y0):int(y1), int(x0):int(x1)]
    ys_px, xs_px = np.nonzero(local)
    if len(xs_px) < 20:
        return False

    pts = np.column_stack([xs_px, ys_px]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    short_side, long_side = min(w, h), max(w, h)
    if short_side == 0:
        return False

    aspect = long_side / short_side
    width_frac = short_side / mask.shape[1]

    return aspect >= min_aspect and width_frac <= max_width_frac

def _road_roi_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    bottom_y = h
    top_y = int(h * 0.45)
    pts = np.array([
        (int(w * 0.05), bottom_y),
        (int(w * 0.38), top_y),
        (int(w * 0.62), top_y),
        (int(w * 0.95), bottom_y),
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _group_max_residual(segs):
    pts = np.array(
        [[(s[0] + s[2]) / 2, (s[1] + s[3]) / 2] for s in segs], dtype=np.float32
    )
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    point, normal = np.array([x0, y0]), np.array([-vy, vx])
    return max(abs(np.dot(p - point, normal)) for p in pts)


def _is_coherent_line(segs, img_width, max_residual_frac=0.015):
    """A real lane line stays close to one straight line even when dashed
    and sweeping toward the vanishing point. This is the straightness check, run after
    grouping."""
    if len(segs) < 2:
        return True
    return _group_max_residual(segs) <= max_residual_frac * img_width


def _analyze_color_channel(mask, img_width, min_total_segments=4):
    mask = _clean_mask(mask)
    segments = _line_segments_from_mask(mask)

    if len(segments) < min_total_segments:
        return None

    groups = _group_by_angle_and_offset(segments)
    groups = [
        g for g in groups
        if _is_coherent_line(g, img_width) and _stripe_geometry(mask, g)
    ]
    if not groups:
        return None

    primary_group = max(groups, key=len)
    pattern, fill_ratio = _classify_pattern(primary_group)

    # a real line rarely goes past ~1.2 here (Hough sometimes catches
    # both edges of one stroke as separate detections) -- past that it's
    # still texture even after the collinearity filter above
    if fill_ratio > 1.2:
        return None

    is_double = _detect_doubles(groups, img_width)
    # confidence comes from the winning group's own segment count, not
    # the total across every group in the mask -- used to inflate
    # confidence off unrelated texture elsewhere in frame
    confidence = min(1.0, len(primary_group) / 25.0)

    return {
        "pattern": pattern,
        "fill_ratio": round(float(fill_ratio), 2),
        "double_line": is_double,
        "segment_count": len(segments),
        "group_count": len(groups),
        "confidence": round(float(confidence), 2),
    }


def detect_road_markings(image: Image.Image):
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]

    white_mask, yellow_mask = _color_masks(img)
    roi_mask = _road_roi_mask(img.shape)
    white_mask = cv2.bitwise_and(white_mask, roi_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask)

    white_result = _analyze_color_channel(white_mask, w)
    yellow_result = _analyze_color_channel(yellow_mask, w)

    return {
        "white_line": white_result,
        "yellow_line": yellow_result,
        "any_markings_detected": white_result is not None or yellow_result is not None,
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_road_markings(road_data):
    if not road_data.get("any_markings_detected"):
        return "No clear lane markings detected — try a closer or less obstructed crop of the road."

    clues = []

    white = road_data.get("white_line")
    if white:
        hedge = "" if white["confidence"] > 0.5 else " (low confidence)"
        clues.append(f"{'Double' if white['double_line'] else 'Single'} {white['pattern']} white line{hedge}")

    yellow = road_data.get("yellow_line")
    if yellow:
        hedge = "" if yellow["confidence"] > 0.5 else " (low confidence)"
        clues.append(f"{'Double' if yellow['double_line'] else 'Single'} {yellow['pattern']} yellow line{hedge}")
        clues.append(
            "Yellow centerlines are common in the Americas and parts of Asia; "
            "most of Europe, Africa, and Oceania use white centerlines instead"
        )

    return " · ".join(clues)


# ---------------------------------------------------------------------------
# Entry point for app.py
# ---------------------------------------------------------------------------

def analyze_road_markings(image: Image.Image):
    """Detection + interpretation in one call, so app.py just imports this."""
    raw = detect_road_markings(image)
    interpreted = interpret_road_markings(raw)
    print("[ROAD DEBUG] white:", raw["white_line"], flush=True)
    print("[ROAD DEBUG] yellow:", raw["yellow_line"], flush=True)
    return raw, interpreted