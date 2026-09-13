"""
Color Service Module
Extracts average RGB values from egg yolk images using Center Circular Masking (42% radius)
and converts RGB to CIELAB (L*, a*, b*) color space according to the CIE D65 standard.
Provides OpenCV-based auto-cropping for yolk regions.
"""

import io
import os
import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


class ColorFeatureList(list):
    """
    List subclass containing [mean_R, mean_G, mean_B, L*, a*, b*]
    with support for dictionary-style key lookups ('r', 'g', 'b', 'l', 'a', 'b_lab').
    """
    _KEY_MAP = {
        'r': 0, 'mean_r': 0, 'R': 0,
        'g': 1, 'mean_g': 1, 'G': 1,
        'b': 2, 'mean_b': 2, 'B': 2,
        'l': 3, 'L': 3, 'l*': 3, 'L*': 3,
        'a': 4, 'A': 4, 'a*': 4, 'A*': 4,
        'b_lab': 5, 'b*': 5, 'B*': 5,
    }

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self._KEY_MAP:
                return super().__getitem__(self._KEY_MAP[item])
            raise KeyError(f"Key '{item}' not recognized in ColorFeatureList")
        return super().__getitem__(item)


def auto_crop_yolk(image_input):
    """
    Auto-crop egg yolk region using OpenCV HSV color masking.
    
    1. Converts BGR -> HSV
    2. Applies cv2.inRange with lower=[10,80,80] and upper=[28,255,250]
    3. Applies morphological close + open (kernel 7x7) to clean the mask
    4. Finds the largest contour with cv2.findContours
    5. Gets the enclosing circle with cv2.minEnclosingCircle
    6. Crops a square region: (cx - r*1.08, cy - r*1.08) to (cx + r*1.08, cy + r*1.08)
    
    :param image_input: str/Path (file path), numpy.ndarray (BGR), or PIL.Image
    :return: tuple (cropped_pil_image, (cx, cy, radius))
    """
    if isinstance(image_input, (str, os.PathLike)):
        bgr = cv2.imread(str(image_input))
        if bgr is None:
            raise FileNotFoundError(f"Could not load image from: {image_input}")
    elif isinstance(image_input, np.ndarray):
        bgr = image_input.copy()
    elif isinstance(image_input, Image.Image):
        # Convert PIL RGB to OpenCV BGR
        rgb_arr = np.array(image_input.convert('RGB'))
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    h, w = bgr.shape[:2]

    # 1. BGR -> HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 2. InRange thresholding: Hue 10-28, Sat 80-255, Val 80-250
    lower_hsv = np.array([10, 80, 80], dtype=np.uint8)
    upper_hsv = np.array([28, 255, 250], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 3. Morphological close + open (kernel 7x7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # 5. Min enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
    else:
        # Fallback: center circle
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) * 0.40

    # 6. Crop square region: (cx ± r*1.08, cy ± r*1.08)
    crop_size = radius * 1.08
    x1 = max(0, int(round(cx - crop_size)))
    y1 = max(0, int(round(cy - crop_size)))
    x2 = min(w, int(round(cx + crop_size)))
    y2 = min(h, int(round(cy + crop_size)))

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h

    cropped_bgr = bgr[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    return cropped_pil, (float(cx), float(cy), float(radius))


def extract_mean_rgb(image_input):
    """
    Extract average R, G, B values from pure yolk region using Center Circular Masking (R=42%).
    
    :param image_input: str/Path, bytes, or PIL.Image object
    :return: dict with 'r', 'g', 'b' (rounded floats 0-255)
    """
    if isinstance(image_input, (str, os.PathLike)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise ValueError("Unsupported image input type.")

    img_rgb = img.convert('RGB')
    np_img = np.array(img_rgb)

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
    Convert RGB values (0-255) to CIELAB color space (L*, a*, b*) using CIE D65 standard.
    
    :param r: Red component (0-255)
    :param g: Green component (0-255)
    :param b: Blue component (0-255)
    :return: dict with 'l', 'a', 'b'
    """
    rgb_norm = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float64)
    lab_arr = rgb2lab(rgb_norm)[0, 0]

    return {
        'l': round(float(lab_arr[0]), 2),
        'a': round(float(lab_arr[1]), 2),
        'b': round(float(lab_arr[2]), 2)
    }


def extract_color_features(image_input):
    """
    Extract color features from an image using Center Circular Mask at radius = min(w,h) * 0.42.
    Returns [mean_R, mean_G, mean_B, L*, a*, b*] (CIE D65 standard).
    
    :param image_input: PIL.Image, str/Path, bytes, or numpy.ndarray
    :return: ColorFeatureList containing [mean_R, mean_G, mean_B, L*, a*, b*]
    """
    rgb = extract_mean_rgb(image_input)
    lab = rgb_to_cielab(rgb['r'], rgb['g'], rgb['b'])
    return ColorFeatureList([
        rgb['r'],
        rgb['g'],
        rgb['b'],
        lab['l'],
        lab['a'],
        lab['b']
    ])


if __name__ == '__main__':
    test_img = Image.new('RGB', (100, 100), color=(226, 131, 23))
    feats = extract_color_features(test_img)
    print("Test features list:", feats)
    print("Test indexing feats[0]:", feats[0])
    print("Test dict access feats['r']:", feats['r'])
    print("Test dict access feats['b_lab']:", feats['b_lab'])
