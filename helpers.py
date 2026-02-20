"""
Helper functions for bounding box metrics and fallen package detection.
"""


def calculate_bbox_metrics(x1, y1, x2, y2):
    """Return (area, aspect_ratio) for a bounding box."""
    area = (x2 - x1) * (y2 - y1)
    aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
    return area, aspect_ratio


def detect_fallen_package(current_bbox, prev_bbox,
                          area_threshold=0.35, aspect_threshold=0.40):
    """
    Detect if a package has fallen by comparing current vs previous bbox.
    Returns True if the area shrinks significantly or aspect ratio changes abruptly.
    """
    if prev_bbox is None:
        return False

    x1_c, y1_c, x2_c, y2_c = current_bbox
    x1_p, y1_p, x2_p, y2_p = prev_bbox

    curr_area, curr_aspect = calculate_bbox_metrics(x1_c, y1_c, x2_c, y2_c)
    prev_area, prev_aspect = calculate_bbox_metrics(x1_p, y1_p, x2_p, y2_p)

    if prev_area == 0:
        return False

    area_change = (prev_area - curr_area) / prev_area
    is_area_shrinking = area_change > area_threshold

    if prev_aspect > 0:
        aspect_change = abs(curr_aspect - prev_aspect) / prev_aspect
        is_aspect_changing = aspect_change > aspect_threshold
    else:
        is_aspect_changing = False

    return is_area_shrinking or is_aspect_changing
