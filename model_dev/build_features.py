"""
Build Features Script
Extracts RGB and CIELAB color features from the auto-cropped dataset (data/auto_cropped_dataset)
using Center Circular Masking (42% radius) and saves results to data/features.csv.
"""

import os
import pandas as pd
from color_service import extract_color_features


def build_features():
    base_dev_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dev_dir, 'data')
    auto_cropped_dir = os.path.join(data_dir, 'auto_cropped_dataset')

    # Fallback to raw_images if auto_cropped_dataset not yet built
    dataset_dir = auto_cropped_dir if os.path.exists(auto_cropped_dir) else os.path.join(data_dir, 'raw_images')

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return None

    class_folders = sorted(
        [d for d in os.listdir(dataset_dir) if d.startswith('class') and os.path.isdir(os.path.join(dataset_dir, d))],
        key=lambda x: int(x.replace('class', ''))
    )

    labels_data = []
    features_data = []
    total_images = 0

    print(f"Extracting color features from: {dataset_dir}")

    for class_folder in class_folders:
        fan_score = int(class_folder.replace('class', ''))
        folder_path = os.path.join(dataset_dir, class_folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Processing {class_folder} ({len(image_files)} images)...", end=" ", flush=True)

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)

            # NOTE: labels_data.append is inside the try block intentionally.
            # This ensures labels.csv and features.csv always have the same number of rows.
            # If feature extraction fails, the image is skipped from BOTH CSVs to prevent
            # row-count mismatch that would corrupt joins/merges.
            try:
                features = extract_color_features(img_path)
                labels_data.append({
                    'image_filename': f"{class_folder}_{img_name}",
                    'original_class': class_folder,
                    'fan_score': fan_score
                })
                features_data.append({
                    'image_filename': f"{class_folder}_{img_name}",
                    'fan_score': fan_score,
                    'r': features['r'],
                    'g': features['g'],
                    'b': features['b'],
                    'l': features['l'],
                    'a': features['a'],
                    'b_lab': features['b_lab']
                })
                total_images += 1
            except Exception as e:
                print(f"Skipping {img_name} (error: {e})")
                continue

        print("Done!")

    # Save to CSV
    labels_df = pd.DataFrame(labels_data)
    labels_csv_path = os.path.join(data_dir, 'labels.csv')
    labels_df.to_csv(labels_csv_path, index=False)

    features_df = pd.DataFrame(features_data)
    features_csv_path = os.path.join(data_dir, 'features.csv')
    features_df.to_csv(features_csv_path, index=False)

    print(f"\nSuccessfully extracted features for {total_images} images!")
    print(f"Saved to: {features_csv_path}")

    return features_df


if __name__ == '__main__':
    build_features()
