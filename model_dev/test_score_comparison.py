"""
Test D: Full Pipeline Score Comparison
Verifies that 20 images run through:
  Pipeline 1: OpenCV crop -> SVR model -> score
  Pipeline 2: Dart-mirror crop -> SVR model -> score
yield predicted DSM Fan scores differing by less than 0.1.
"""

import os
import sys
import joblib
import cv2
import numpy as np
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure color_service can be imported
sys.path.insert(0, os.path.dirname(__file__))
from color_service import auto_crop_yolk, extract_color_features


def pure_dart_mirror_crop(image_path):
    """
    Pure arithmetic yolk detection and cropping mirroring Dart YolkDetector.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

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

    cv_h = np.round(h_deg / 2.0)
    cv_s = np.round(s * 255.0)
    cv_v = np.round(v)

    mask = (
        (cv_h >= 10) & (cv_h <= 28) &
        (cv_s >= 80) & (cv_s <= 255) &
        (cv_v >= 80) & (cv_v <= 250)
    )

    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) * 0.40
    else:
        cx = float(np.median(x_indices))
        cy = float(np.median(y_indices))
        dists = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        radius = float(np.percentile(dists, 99.5))

    crop_size = radius * 1.08
    x1 = max(0, int(round(cx - crop_size)))
    y1 = max(0, int(round(cy - crop_size)))
    x2 = min(w, int(round(cx + crop_size)))
    y2 = min(h, int(round(cy + crop_size)))

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h

    cropped_rgb = rgb[y1:y2, x1:x2]
    return Image.fromarray(cropped_rgb), (cx, cy, radius)


def get_20_test_images():
    search_dirs = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'pic_egg_yolk'),
        os.path.join(os.path.dirname(__file__), 'data', 'raw_images'),
    ]
    img_paths = []
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for root, dirs, files in os.walk(sdir):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_paths.append(os.path.join(root, f))
                        if len(img_paths) == 20:
                            return img_paths
    if len(img_paths) < 20:
        raise FileNotFoundError(f"Could not find 20 test images in {search_dirs}")
    return img_paths


def run_test():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    test_images = get_20_test_images()

    print("Full Pipeline Score Comparison (OpenCV Crop vs Dart-Mirror Crop):")
    print(f"{'img':<6} {'OpenCV_score':<14} {'DartMirror_score':<18} {'diff':<6}")
    print("-" * 50)

    all_pass = True
    for img_path in test_images:
        # 1. OpenCV pipeline
        cv_crop_img, _ = auto_crop_yolk(img_path)
        cv_feats = extract_color_features(cv_crop_img)
        cv_vec = np.array([[
            cv_feats['r'], cv_feats['g'], cv_feats['b'],
            cv_feats['l'], cv_feats['a'], cv_feats['b_lab']
        ]])
        score_cv = float(pipeline.predict(cv_vec)[0])

        # 2. Pure Dart-mirror pipeline
        dart_crop_img, _ = pure_dart_mirror_crop(img_path)
        dart_feats = extract_color_features(dart_crop_img)
        dart_vec = np.array([[
            dart_feats['r'], dart_feats['g'], dart_feats['b'],
            dart_feats['l'], dart_feats['a'], dart_feats['b_lab']
        ]])
        score_dart = float(pipeline.predict(dart_vec)[0])

        diff = abs(score_cv - score_dart)
        passed = diff < 0.1
        if not passed:
            all_pass = False

        status = "✓" if passed else "✗"
        img_name = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')[:5]

        print(f"{img_name:<6} {score_cv:<14.1f} {score_dart:<18.1f} {diff:<6.1f} {status}")

    print("-" * 50)
    assert all_pass, "Some score predictions differed by >= 0.1"
    print("...all pass ✓")


if __name__ == '__main__':
    run_test()
