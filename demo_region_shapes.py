"""Visual demo: apply the same Ch/Cl frequency-perturbation formulas to a
real photo, once using a rectangular region (existing FCM) and once using a
radial region (new RadialOriginalFreqTune), side by side. No training
needed -- just run it on any photo and open the saved PNG.

Usage:
    python3 demo_region_shapes.py my_photo.jpg
    python3 demo_region_shapes.py my_photo.jpg -o out.png --seed 3
"""
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rect_perturb(img, rng):
    height, width = img.shape[:2]
    fft_1 = np.fft.fftn(img)

    x_min = rng.randint(width // 32 + 1, width // 2)
    x_max = rng.randint(width // 2, width - width // 32 - 1)
    y_min = rng.randint(height // 32 + 1, height // 2)
    y_max = rng.randint(height // 2, height - height // 32 - 1)
    matrix = fft_1[x_min:x_max, y_min:y_max]

    B, A = 0.5, 5
    b = rng.uniform(0, B)
    array2 = rng.uniform(1 - b, 1 + b, size=fft_1.shape)
    a = rng.uniform(0, A)
    array1 = rng.uniform(-a, a, size=matrix.shape)

    fft_1 = fft_1 * array2
    fft_1[x_min:x_max, y_min:y_max] = matrix * array1

    out = np.fft.ifftn(fft_1)
    return np.clip(np.real(out), 0, 255).astype(np.uint8)


def radial_perturb(img, rng):
    height, width = img.shape[:2]
    fft_1 = np.fft.fftn(img)
    shifted = np.fft.fftshift(fft_1, axes=(0, 1))

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    dist = np.sqrt((xx - width // 2) ** 2 + (yy - height // 2) ** 2)[:, :, None]
    max_r = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    r = rng.uniform(0, max_r)
    high_mask = dist >= r

    B, A = 0.5, 5
    b = rng.uniform(0, B)
    array_low = rng.uniform(1 - b, 1 + b, size=shifted.shape)
    a = rng.uniform(0, A)
    array_high = rng.uniform(-a, a, size=shifted.shape)

    perturb = np.where(high_mask, array_high, array_low)
    shifted = shifted * perturb
    fft_1 = np.fft.ifftshift(shifted, axes=(0, 1))

    out = np.fft.ifftn(fft_1)
    return np.clip(np.real(out), 0, 255).astype(np.uint8)


def main():
    argv = sys.argv[1:]
    out_path = 'region_shapes_demo.png'
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
        print('Usage: python3 demo_region_shapes.py <photo_path> [-o out.png] [--seed N]')
        sys.exit(1)

    img = np.array(Image.open(argv[0]).convert('RGB').resize((256, 256)))

    rect = rect_perturb(img, np.random.RandomState(seed))
    radial = radial_perturb(img, np.random.RandomState(seed))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[1].imshow(rect)
    axes[1].set_title('Rectangular region\n(existing FCM)')
    axes[2].imshow(radial)
    axes[2].set_title('Radial region\n(new RadialFreqTune)')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
