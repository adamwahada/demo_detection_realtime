"""
Helper functions for bounding box metrics.
"""


def calculate_bbox_metrics(x1, y1, x2, y2):
    """Return (area, aspect_ratio) for a bounding box."""
    area = (x2 - x1) * (y2 - y1)
    aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
    return area, aspect_ratio
