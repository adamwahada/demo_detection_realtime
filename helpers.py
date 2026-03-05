"""
Helper functions for bounding box metrics.
"""


def calculate_bbox_metrics(x1, y1, x2, y2):
    """Return (area, aspect_ratio) for a bounding box."""
    area = (x2 - x1) * (y2 - y1)
    aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
    return area, aspect_ratio

def letterbox_image(image, size=640):
    """Resizes image to a square size using padding (letterboxing)."""
    if isinstance(size, int):
        size = (size, size)
        
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    if nw < iw or nh < ih:
        interp = cv2.INTER_AREA   # best for downsampling
    else:
        interp = cv2.INTER_CUBIC  # best for upsampling

    image = cv2.resize(image, (nw, nh), interpolation=interp)
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Center the image
    top = (h - nh) // 2
    left = (w - nw) // 2
    new_image[top:top+nh, left:left+nw, :] = image
    return new_image