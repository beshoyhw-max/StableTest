"""
Unicode-safe text rendering for OpenCV.

cv2.putText only supports ASCII characters. Chinese/CJK characters render as "????".
This module provides a drop-in replacement that uses PIL for non-ASCII text.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Cache fonts to avoid reloading on every call
_font_cache = {}


def _get_font(size):
    """Get a cached font that supports CJK characters."""
    if size in _font_cache:
        return _font_cache[size]

    font = None
    # Try Windows CJK fonts (most common on Chinese Windows)
    win_fonts = [
        "msyh.ttc",      # Microsoft YaHei (微软雅黑)
        "simhei.ttf",    # SimHei (黑体)
        "simsun.ttc",    # SimSun (宋体)
        "msyhbd.ttc",    # Microsoft YaHei Bold
        "arial.ttf",     # Fallback
    ]

    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")

    for font_name in win_fonts:
        font_path = os.path.join(fonts_dir, font_name)
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                break
            except Exception:
                continue

    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()

    _font_cache[size] = font
    return font


def _has_non_ascii(text):
    """Check if text contains any non-ASCII characters."""
    try:
        text.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def put_text(img, text, org, font_face, font_scale, color, thickness, line_type=None):
    """
    Drop-in replacement for cv2.putText that supports Unicode/CJK characters.
    
    Uses PIL rendering when text contains non-ASCII chars, otherwise falls back
    to cv2.putText for maximum performance.
    
    Args: Same as cv2.putText (font_face and line_type are ignored for PIL path)
    """
    if not _has_non_ascii(text):
        # Fast path: pure ASCII, use native OpenCV
        if line_type is not None:
            cv2.putText(img, text, org, font_face, font_scale, color, thickness, line_type)
        else:
            cv2.putText(img, text, org, font_face, font_scale, color, thickness)
        return

    # PIL path for Unicode text
    # Convert font_scale to approximate pixel size (cv2 scale 1.0 ≈ 22px)
    font_size = max(16, int(font_scale * 32))
    font = _get_font(font_size)

    # Convert BGR color to RGB for PIL
    if len(color) == 3:
        rgb_color = (int(color[2]), int(color[1]), int(color[0]))
    else:
        rgb_color = (int(color[2]), int(color[1]), int(color[0]), int(color[3]))

    # Convert OpenCV image (BGR) to PIL Image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # cv2.putText origin is bottom-left of text baseline
    # PIL uses top-left corner, so we need to adjust
    bbox = draw.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]

    x, y = org
    # Shift up by text height (cv2 baseline -> PIL top-left)
    pil_y = y - text_height

    draw.text((x, pil_y), text, font=font, fill=rgb_color,
              stroke_width=max(1, thickness - 1), stroke_fill=rgb_color)

    # Convert back to OpenCV BGR
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)


def get_text_size(text, font_face, font_scale, thickness):
    """
    Drop-in replacement for cv2.getTextSize that supports Unicode/CJK characters.
    
    Returns: ((width, height), baseline) - same format as cv2.getTextSize
    """
    if not _has_non_ascii(text):
        return cv2.getTextSize(text, font_face, font_scale, thickness)

    font_size = max(16, int(font_scale * 32))
    font = _get_font(font_size)

    # Use PIL to measure text
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    baseline = max(1, int(height * 0.2))

    return (width, height), baseline
