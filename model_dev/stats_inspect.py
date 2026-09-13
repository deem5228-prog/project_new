import os
import glob
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv

base_dir = r'c:\Users\Admin\Downloads\project_code\pic_egg_yolk'
classes = [f"class{i}" for i in range(4, 16)]

stats_by_class = {}
all_bg_sats = []
all_yolk_sats = []
all_yolk_hues = []
all_yolk_vals = []

for c in classes:
    folder = os.path.join(base_dir, c)
    if not os.path.exists(folder):
        continue
    files = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
    hues, sats, vals = [], [], []
    bg_sats = []
    
    for f in files:
        img = Image.open(f).convert('RGB')
        arr = np.array(img) / 255.0
        hsv = rgb2hsv(arr)
        h = hsv[:, :, 0] * 360.0
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        H, W, _ = arr.shape
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        cyolk_s = s[cy-r:cy+r, cx-r:cx+r]
        cyolk_h = h[cy-r:cy+r, cx-r:cx+r]
        cyolk_v = v[cy-r:cy+r, cx-r:cx+r]
        
        s_med = np.median(cyolk_s)
        h_med = np.median(cyolk_h)
        v_med = np.median(cyolk_v)
        
        sats.append(s_med)
        hues.append(h_med)
        vals.append(v_med)
        all_yolk_sats.append(s_med)
        all_yolk_hues.append(h_med)
        all_yolk_vals.append(v_med)
        
        # Corner background (4 corners)
        corner_bg = [
            s[0:15, 0:15],
            s[0:15, -15:],
            s[-15:, 0:15],
            s[-15:, -15:]
        ]
        bg_mean = np.mean([np.mean(cr) for cr in corner_bg])
        bg_sats.append(bg_mean)
        all_bg_sats.append(bg_mean)
        
    stats_by_class[c] = {
        'count': len(files),
        'hue_mean': float(np.mean(hues)),
        'hue_min': float(np.min(hues)),
        'hue_max': float(np.max(hues)),
        'sat_mean': float(np.mean(sats)),
        'sat_min': float(np.min(sats)),
        'sat_max': float(np.max(sats)),
        'val_mean': float(np.mean(vals)),
        'val_min': float(np.min(vals)),
        'bg_sat_mean': float(np.mean(bg_sats)),
        'bg_sat_max': float(np.max(bg_sats))
    }

print(f"{'Class':<8} | {'Count':<5} | {'Hue Mean (Min-Max)':<22} | {'Sat Mean (Min-Max)':<22} | {'Val Mean':<8} | {'BG Sat Max':<10}")
print("-" * 85)
for c in classes:
    st = stats_by_class[c]
    hue_str = f"{st['hue_mean']:.1f} ({st['hue_min']:.1f}-{st['hue_max']:.1f})"
    sat_str = f"{st['sat_mean']:.3f} ({st['sat_min']:.3f}-{st['sat_max']:.3f})"
    print(f"{c:<8} | {st['count']:<5} | {hue_str:<22} | {sat_str:<22} | {st['val_mean']:.2f}     | {st['bg_sat_max']:.3f}")

print("-" * 85)
print(f"Overall Yolk Hue: {np.min(all_yolk_hues):.1f} to {np.max(all_yolk_hues):.1f} (Mean: {np.mean(all_yolk_hues):.1f})")
print(f"Overall Yolk Sat: {np.min(all_yolk_sats):.3f} to {np.max(all_yolk_sats):.3f} (Mean: {np.mean(all_yolk_sats):.3f})")
print(f"Overall Yolk Val: {np.min(all_yolk_vals):.2f} to {np.max(all_yolk_vals):.2f} (Mean: {np.mean(all_yolk_vals):.2f})")
print(f"Overall Corner BG Saturation: Mean {np.mean(all_bg_sats):.3f}, Max {np.max(all_bg_sats):.3f}")
