"""
Test A: HSV Equivalence Check
Verifies that the OpenCV inRange binary mask and the pure Python formula
mirroring Dart YolkDetector match 100.0% pixel-by-pixel.
"""

import os
import sys
import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def find_sample_image():
  candidates = [
      os.path.join(os.path.dirname(__file__), '..', '..', 'pic_egg_yolk', 'class10', '100.jpg'),
      os.path.join(os.path.dirname(__file__), 'data', 'raw_images'),
  ]
  for c in candidates:
    if os.path.isfile(c):
      return c
    if os.path.isdir(c):
      for root, dirs, files in os.walk(c):
        for f in files:
          if f.lower().endswith(('.jpg', '.png')):
            return os.path.join(root, f)
  raise FileNotFoundError("Could not find sample image for Test A")


def run_test():
  img_path = find_sample_image()
  bgr = cv2.imread(img_path)
  if bgr is None:
    raise FileNotFoundError(f"Failed to read image: {img_path}")

  # 1. OpenCV HSV and Mask
  # lower = [10, 80, 80], upper = [28, 255, 250]
  hsv_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  lower_cv = np.array([10, 80, 80], dtype=np.uint8)
  upper_cv = np.array([28, 255, 250], dtype=np.uint8)
  mask_cv = (cv2.inRange(hsv_cv, lower_cv, upper_cv) > 0)

  # 2. Pure Python formula mirroring Dart YolkDetector
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  h_img, w_img, _ = rgb.shape

  r = rgb[:, :, 0].astype(float)
  g = rgb[:, :, 1].astype(float)
  b = rgb[:, :, 2].astype(float)

  v = np.maximum(np.maximum(r, g), b)
  delta = v - np.minimum(np.minimum(r, g), b)
  s = np.where(v > 0, delta / v, 0.0)

  h_deg = np.zeros_like(v)
  mr = (v == r) & (delta > 0)
  mg = (v == g) & (delta > 0) & ~mr
  mb = (v == b) & (delta > 0) & ~mr & ~mg
  h_deg[mr] = (60.0 * (g[mr] - b[mr]) / delta[mr]) % 360.0
  h_deg[mg] = (60.0 * (b[mg] - r[mg]) / delta[mg]) + 120.0
  h_deg[mb] = (60.0 * (r[mb] - g[mb]) / delta[mb]) + 240.0

  # Dart equivalence logic matching OpenCV integer scale:
  cv_h = np.round(h_deg / 2.0)
  cv_s = np.round(s * 255.0)
  cv_v = np.round(v)

  mask_dart = (
      (cv_h >= 10) & (cv_h <= 28) &
      (cv_s >= 80) & (cv_s <= 255) &
      (cv_v >= 80) & (cv_v <= 250)
  )

  # 3. Check binary mask match
  total_pixels = h_img * w_img
  matching = int(np.sum(mask_cv == mask_dart))
  mismatch = total_pixels - matching
  pct = (matching / total_pixels) * 100.0

  print(f"Total pixels: {total_pixels}")
  print(f"Mask match: {matching}/{total_pixels} ({pct:.1f}%)")
  print(f"Mismatch count: {mismatch}")

  assert mismatch == 0, f"Expected 0 mismatches, got {mismatch}"
  print("✓ Test A Passed!")


if __name__ == '__main__':
  run_test()
