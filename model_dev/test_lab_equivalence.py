"""
Test B: CIELAB Equivalence Check
Verifies that CIELAB conversion in the pure formula mirroring Dart ColorExtractor
matches scikit-image (CIE D65 standard) with difference < 0.001 across 10 sample RGB values.
"""

import sys
import numpy as np
from skimage.color import rgb2lab

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def dart_mirror_rgb_to_cielab(r: float, g: float, b: float):
    """
    Pure arithmetic formula mirroring Dart ColorExtractor.rgbToCielab.
    Uses CIE D65 reference white point (Xn=0.95047, Yn=1.00000, Zn=1.08883)
    and standard sRGB gamma decoding.
    """
    # 1. sRGB gamma correction to linear RGB
    r_lin = r / 255.0
    g_lin = g / 255.0
    b_lin = b / 255.0

    r_lin = ((r_lin + 0.055) / 1.055)**2.4 if r_lin > 0.04045 else r_lin / 12.92
    g_lin = ((g_lin + 0.055) / 1.055)**2.4 if g_lin > 0.04045 else g_lin / 12.92
    b_lin = ((b_lin + 0.055) / 1.055)**2.4 if b_lin > 0.04045 else b_lin / 12.92

    # 2. Linear RGB to XYZ (sRGB D65 matrix matching scikit-image)
    x = 0.412453 * r_lin + 0.357580 * g_lin + 0.180423 * b_lin
    y = 0.212671 * r_lin + 0.715160 * g_lin + 0.072169 * b_lin
    z = 0.019334 * r_lin + 0.119193 * g_lin + 0.950227 * b_lin

    # 3. XYZ to CIELAB
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    xr = x / xn
    yr = y / yn
    zr = z / zn

    fx = xr**(1.0 / 3.0) if xr > 0.008856 else (7.787 * xr + 16.0 / 116.0)
    fy = yr**(1.0 / 3.0) if yr > 0.008856 else (7.787 * yr + 16.0 / 116.0)
    fz = zr**(1.0 / 3.0) if zr > 0.008856 else (7.787 * zr + 16.0 / 116.0)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    lab_b = 200.0 * (fy - fz)

    return l, a, lab_b


def run_test():
    # 10 representative RGB values spanning the DSM yolk color range (score 1 to 15)
    sample_rgbs = [
        (255, 220, 80),   # Pale Yellow (Fan 1-3)
        (255, 205, 60),   # Light Yellow (Fan 4-5)
        (250, 190, 45),   # Yellow (Fan 6-7)
        (245, 175, 35),   # Golden Yellow (Fan 8)
        (240, 160, 30),   # Medium Orange (Fan 9-10)
        (235, 145, 25),   # Orange (Fan 11)
        (225, 130, 20),   # Deep Orange (Fan 12-13)
        (215, 115, 18),   # Reddish Orange (Fan 14)
        (205, 95, 15),    # Deep Reddish Orange (Fan 15)
        (255, 180, 50),   # Reference Benchmark from prompt
    ]

    all_pass = True
    print("CIELAB Equivalence Check (scikit-image reference vs Dart-mirror):")
    print("-" * 80)

    for rgb in sample_rgbs:
        r, g, b = rgb
        # Reference from scikit-image
        rgb_norm = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float64)
        ref_lab = rgb2lab(rgb_norm)[0, 0]
        ref_l, ref_a, ref_b = ref_lab[0], ref_lab[1], ref_lab[2]

        # Mine from pure Python formula mirroring Dart
        my_l, my_a, my_b = dart_mirror_rgb_to_cielab(r, g, b)

        diff = max(abs(ref_l - my_l), abs(ref_a - my_a), abs(ref_b - my_b))
        passed = diff < 0.001
        if not passed:
            all_pass = False

        status = "✓" if passed else "✗"
        scikit_str = f"[{ref_l:.1f}, {ref_a:.1f}, {ref_b:.1f}]"
        mine_str = f"[{my_l:.1f}, {my_a:.1f}, {my_b:.1f}]"
        print(f"RGB{rgb}: scikit={scikit_str}  mine={mine_str}  diff={diff:.3f} {status}")

    print("-" * 80)
    assert all_pass, "Some CIELAB equivalence checks failed (diff >= 0.001)"
    print("✓ All CIELAB tests passed (diff < 0.001 across all samples)!")


if __name__ == '__main__':
    run_test()
