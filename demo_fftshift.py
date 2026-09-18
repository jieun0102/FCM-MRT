"""Visual demo: what fftshift does to an FFT magnitude spectrum, and proof
that the fftshift -> ifftshift round trip is lossless. No GPU/training
needed -- just run it and open the saved PNG.

Usage:
    python3 demo_fftshift.py                # uses a built-in synthetic image
    python3 demo_fftshift.py my_photo.jpg   # uses your own image instead
    python3 demo_fftshift.py my_photo.jpg -o out.png
"""
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_demo_image(size=128):
    # Synthetic image with both smooth (low-frequency) and sharp (high-
    # frequency) content, so both regions of the spectrum are visible.
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    gradient = (xx / size) * 255  # smooth left-to-right gradient -> low freq
    cx, cy = size // 2, size // 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = ((r > size * 0.15) & (r < size * 0.18)).astype(np.float64) * 255  # sharp edge -> high freq
    checker = (((xx // 4) + (yy // 4)) % 2) * 60  # fine texture -> high freq
    img = np.clip(gradient * 0.5 + ring + checker * 0.3, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def main():
    out_path = 'fftshift_demo.png'
    argv = sys.argv[1:]
    if '-o' in argv:
        idx = argv.index('-o')
        out_path = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]

    if argv:
        img = np.array(Image.open(argv[0]).convert('RGB').resize((128, 128)))
    else:
        img = make_demo_image()

    gray = img.mean(axis=2)
    fft = np.fft.fftn(gray)
    mag_unshifted = np.log1p(np.abs(fft))
    mag_shifted = np.log1p(np.abs(np.fft.fftshift(fft)))

    # Round-trip check: fftshift -> ifftshift -> ifft should exactly
    # reconstruct the original image (proves the shift loses no information).
    fft_3d = np.fft.fftn(img, axes=(0, 1))
    shifted_3d = np.fft.fftshift(fft_3d, axes=(0, 1))
    restored_3d = np.fft.ifftshift(shifted_3d, axes=(0, 1))
    reconstructed = np.clip(np.real(np.fft.ifftn(restored_3d, axes=(0, 1))), 0, 255).astype(np.uint8)
    max_diff = int(np.abs(reconstructed.astype(int) - img.astype(int)).max())

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original image')
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title('fftshift -> ifftshift -> ifft\n(max pixel diff: %d)' % max_diff)
    axes[1, 0].imshow(mag_unshifted, cmap='viridis')
    axes[1, 0].set_title('FFT magnitude, no shift\nDC sits at the corner [0,0]')
    axes[1, 1].imshow(mag_shifted, cmap='viridis')
    axes[1, 1].set_title('FFT magnitude, fftshift applied\nDC sits at the center')
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved to', out_path)
    print('Round-trip max pixel diff (should be 0, up to rounding):', max_diff)


if __name__ == '__main__':
    main()
