"""
Color Service Module (FastAPI Backend)
Extracts average RGB values from egg yolk images using Center Circular Masking (R=42%)
and converts RGB to CIELAB (L*, a*, b*) color space, Chroma, and Hue angle.
"""

import io
import math
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def extract_mean_rgb(image_input):
    """
    Extract average R, G, B values from pure yolk region using Center Circular Masking (R=42%).
    
    :param image_input: str/Path (file path), bytes, or PIL.Image object
    :return: dict with 'r', 'g', 'b' (rounded floats 0-255)
    """
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("Unsupported image input type. Expected file path, bytes, or PIL.Image.")

    # Convert to RGB mode (handles RGBA or grayscale)
    img_rgb = img.convert('RGB')
    np_img = np.array(img_rgb)

    # Center Circular Masking (Radius = 42% of min dimension to eliminate 4 background corners)
    h, w, _ = np_img.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.42)
    y_coords, x_coords = np.ogrid[:h, :w]
    mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2

    if not mask.any():
        mean_r = float(np.mean(np_img[:, :, 0]))
        mean_g = float(np.mean(np_img[:, :, 1]))
        mean_b = float(np.mean(np_img[:, :, 2]))
    else:
        mean_r = float(np.mean(np_img[:, :, 0][mask]))
        mean_g = float(np.mean(np_img[:, :, 1][mask]))
        mean_b = float(np.mean(np_img[:, :, 2][mask]))

    return {
        'r': round(mean_r, 2),
        'g': round(mean_g, 2),
        'b': round(mean_b, 2)
    }


def rgb_to_cielab(r: float, g: float, b: float):
    """
    Convert RGB values (0-255) to CIELAB color space (L*, a*, b*).
    
    :param r: Red component (0-255)
    :param g: Green component (0-255)
    :param b: Blue component (0-255)
    :return: dict with 'l', 'a', 'b'
    """
    # Normalize RGB to [0, 1] range for skimage rgb2lab
    rgb_norm = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float64)
    lab_arr = rgb2lab(rgb_norm)[0, 0]

    l_val = float(lab_arr[0])
    a_val = float(lab_arr[1])
    b_val = float(lab_arr[2])

    return {
        'l': round(l_val, 2),
        'a': round(a_val, 2),
        'b': round(b_val, 2)
    }


def calculate_chroma_and_hue(a: float, b: float):
    """
    Calculate Chroma (C*) and Hue Angle (h in degrees) from CIELAB a* and b*.
    """
    chroma = math.sqrt(a**2 + b**2)
    hue_rad = math.atan2(b, a)
    hue_deg = math.degrees(hue_rad)
    if hue_deg < 0:
        hue_deg += 360.0
    return round(chroma, 2), round(hue_deg, 2)


def extract_color_features(image_input):
    """
    Extract both RGB, CIELAB, Chroma, and Hue color features from an image.
    """
    rgb = extract_mean_rgb(image_input)
    lab = rgb_to_cielab(rgb['r'], rgb['g'], rgb['b'])
    chroma, hue = calculate_chroma_and_hue(lab['a'], lab['b'])
    
    return {
        'r': rgb['r'],
        'g': rgb['g'],
        'b': rgb['b'],
        'l': lab['l'],
        'a': lab['a'],
        'b_lab': lab['b'],
        'chroma': chroma,
        'hue': hue,
        'hue_angle': hue
    }
