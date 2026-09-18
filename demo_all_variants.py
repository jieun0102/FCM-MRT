"""Visual demo: apply the actual FCM / FCM-MRT transform classes (from
FreqTune_transform.py, the same code cifar.py trains with) to one photo,
comparing rectangular vs radial region selection for both algorithms.
Runs at native 32x32 -- no training needed.

Usage:
    python3 demo_all_variants.py my_photo.jpg
    python3 demo_all_variants.py my_photo.jpg -o out.png --seed 3
"""
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import FreqTune_transform as ft


def main():
    argv = sys.argv[1:]
    out_path = 'all_variants_demo.png'
    seed = 0
    if '-o' in argv:
        idx = argv.index('-o')
        out_path = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    if '--seed' in argv:
        idx = argv.index('--seed')
        seed = int(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2:]

    if not argv:
        print('Usage: python3 demo_all_variants.py <photo_path> [-o out.png] [--seed N]')
        sys.exit(1)

    img = Image.open(argv[0]).convert('RGB').resize((32, 32))

    variants = [
        ('Original', None),
        ('FCM\n(rectangular)', ft.OriginalFreqTune(probability=1.0)),
        ('FCM\n(radial)', ft.RadialOriginalFreqTune(probability=1.0)),
        ('FCM-MRT\n(rectangular)', ft.TwoRegionFreqTune(probability=1.0, mode='uniform')),
        ('FCM-MRT\n(radial)', ft.RadialFreqTune(probability=1.0, mode='uniform')),
    ]

    fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3.5))
    for ax, (title, transform) in zip(axes, variants):
        np.random.seed(seed)
        out = img if transform is None else transform(img)
        ax.imshow(out)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
