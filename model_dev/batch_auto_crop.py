"""
Batch Auto-Cropping Script
Automatically detects and crops egg yolks from raw images across all 12 classes (class4..class15)
using HSV Color-based Segmentation + Bounding Box Centroid Square Crop.
Also exports normalized YOLO format bounding box txt files for future deep learning research.
"""

import os
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv


def detect_and_crop_yolk(image_path, padding_ratio=0.05):
    """
    Detects yolk in an image using HSV Color Segmentation and crops a square region around it.
    Returns: (cropped_pil_image, yolo_bbox_tuple)
    yolo_bbox_tuple: (x_center_norm, y_center_norm, w_norm, h_norm)

    HSV thresholds are synced with color_service.py (OpenCV equivalent):
      - Hue: 20°–56°  (OpenCV: 10–28 in 0–180 scale → ×2 = 20°–56°)
      - Saturation: ≥ 0.314  (OpenCV: 80/255)
      - Value: 0.314–0.980  (OpenCV: 80/255–250/255)
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    h_img, w_img = arr.shape[:2]

    # Convert RGB to HSV (skimage: H in [0,360], S/V in [0,1])
    hsv = rgb2hsv(arr / 255.0)
    h, s, v = hsv[:, :, 0] * 360.0, hsv[:, :, 1], hsv[:, :, 2]

    # --- M-15 FIX: HSV thresholds synced with color_service.py (OpenCV) ---
    # OpenCV lower=[10,80,80] upper=[28,255,250] in 0-180/0-255/0-255 scale
    # Converted to skimage scale (H: ×2, S/V: ÷255):
    #   Hue: 20°–56°  |  Sat: 80/255≈0.314–1.0  |  Val: 80/255≈0.314–250/255≈0.980
    mask = (
        (h >= 20.0) & (h <= 56.0) &
        (s >= 0.314) &
        (v >= 0.314) & (v <= 0.980)
    )
    matched_count = np.sum(mask)

    # If yolk pixels detected (> 1% of total image)
    if matched_count > (arr.shape[0] * arr.shape[1] * 0.01):
        # Isolate individual color blobs (Separates distractions from actual yolk)
        from skimage.measure import label, regionprops
        lbl = label(mask)
        props = regionprops(lbl)

        if props:
            # Select the LARGEST blob (egg yolk is significantly larger than foreign objects)
            largest = max(props, key=lambda p: p.area)
            min_y, min_x, max_y, max_x = largest.bbox
            cy, cx = int(largest.centroid[0]), int(largest.centroid[1])

            bbox_w = max_x - min_x
            bbox_h = max_y - min_y

            # Normalized YOLO Bounding Box for the original image
            yolo_xc = (min_x + max_x) / 2.0 / w_img
            yolo_yc = (min_y + max_y) / 2.0 / h_img
            yolo_w = bbox_w / w_img
            yolo_h = bbox_h / h_img

            # Make square crop with padding, centered on the yolk centroid
            box_size = int(max(bbox_w, bbox_h) * (1.0 + padding_ratio))
            half = box_size // 2

            # --- M-14 FIX: Re-center crop after clamping to image bounds ---
            # Instead of shifting x1/y1 and losing center alignment,
            # clamp the center and derive x1/y1 from the clamped center.
            cx_clamped = int(np.clip(cx, half, w_img - half))
            cy_clamped = int(np.clip(cy, half, h_img - half))

            x1 = cx_clamped - half
            y1 = cy_clamped - half
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Final safety clamp (handles edge case where box_size > image dimension)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            actual_size = min(x2 - x1, y2 - y1)
            cropped_img = img.crop((x1, y1, x1 + actual_size, y1 + actual_size))

            return cropped_img, (yolo_xc, yolo_yc, yolo_w, yolo_h)

        else:
            # --- C-3 FIX: regionprops is empty despite mask pixels > 1% ---
            # Fall back to center square crop instead of returning None
            # (returning None would crash the caller's tuple-unpack on line 107)
            print(f"    [WARN] regionprops empty for {image_path}, using center crop fallback.")
            min_dim = min(w_img, h_img)
            x1 = (w_img - min_dim) // 2
            y1 = (h_img - min_dim) // 2
            cropped_img = img.crop((x1, y1, x1 + min_dim, y1 + min_dim))
            return cropped_img, (0.5, 0.5, min_dim / w_img, min_dim / h_img)

    else:
        # Fallback to center square crop if color mask fails
        min_dim = min(w_img, h_img)
        x1 = (w_img - min_dim) // 2
        y1 = (h_img - min_dim) // 2
        cropped_img = img.crop((x1, y1, x1 + min_dim, y1 + min_dim))
        return cropped_img, (0.5, 0.5, min_dim / w_img, min_dim / h_img)



def run_batch_auto_crop():
    base_dir = os.path.dirname(__file__)
    source_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'pic_egg_yolk'))
    output_dir = os.path.join(base_dir, 'data', 'auto_cropped_dataset')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading source images from: {source_dir}")
    print(f"Saving auto-cropped dataset to: {output_dir}\n")

    class_folders = sorted(
        [d for d in os.listdir(source_dir) if d.startswith('class') and os.path.isdir(os.path.join(source_dir, d))],
        key=lambda x: int(x.replace('class', ''))
    )

    total_processed = 0

    for class_name in class_folders:
        in_class_dir = os.path.join(source_dir, class_name)
        out_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        images = [f for f in os.listdir(in_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {class_name} ({len(images)} images)...", end=" ", flush=True)

        for img_file in images:
            src_path = os.path.join(in_class_dir, img_file)
            out_img_path = os.path.join(out_class_dir, img_file)
            out_label_path = os.path.join(out_class_dir, os.path.splitext(img_file)[0] + '.txt')

            # Detect, crop and get YOLO bbox
            cropped_img, (yolo_xc, yolo_yc, yolo_w, yolo_h) = detect_and_crop_yolk(src_path)

            # Save cropped image
            cropped_img.save(out_img_path, quality=95)

            # Save YOLO annotation (class 0: egg_yolk)
            with open(out_label_path, 'w', encoding='utf-8') as f:
                f.write(f"0 {yolo_xc:.6f} {yolo_yc:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")

            total_processed += 1

        print("Done!")

    print(f"\nSuccessfully auto-cropped and annotated all {total_processed} images!")
    print(f"Dataset ready at: {output_dir}")


if __name__ == '__main__':
    run_batch_auto_crop()
