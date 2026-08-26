"""Compare an original CIFAR image against a DC-preserved TwoRegionFreqTune output.

FreqTune_transform.py was reverted back to the original algorithm (no
preserve_dc option), so this script keeps that file untouched and instead
re-implements the same TwoRegionFreqTune math here, with an added option to
restore the DC (0,0,0) frequency term after the random perturbations. This
lets us visually check whether fixing the DC term removes the brightness
drift, without changing the shared transform file.

Usage:
    python compare_dc_fix.py --idx 8
"""
import os
import random
import argparse
import pickle

import numpy as np
from PIL import Image


def load_cifar_batch(batch_path):
    with open(batch_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')['data']


def row_to_image(row):
    arr = row.reshape(3, 32, 32).transpose(1, 2, 0)
    return Image.fromarray(arr.astype(np.uint8), mode='RGB')


def two_region_freqtune(x, preserve_dc, seed):
    """Same math as the original TwoRegionFreqTune.__call__, minus x.show()."""
    random.seed(seed)
    np.random.seed(seed)

    height = 32
    width = 32
    img = np.array(x)
    fft_1 = np.fft.fftn(img)

    dc_index = tuple(0 for _ in range(fft_1.ndim))
    original_dc = fft_1[dc_index].copy() if preserve_dc else None

    x_min = np.random.randint(width // 32, width // 2)
    x_max = np.random.randint(width // 2, width - width // 32)
    y_min = np.random.randint(height // 32, height // 2)
    y_max = np.random.randint(height // 2, height - height // 32)

    x_min_2 = np.random.randint(width // 32, x_min + 1)
    x_max_2 = np.random.randint(x_max, width - width // 32)
    y_min_2 = np.random.randint(height // 32, y_min + 1)
    y_max_2 = np.random.randint(y_max, height - height // 32)

    matrix_1 = fft_1[x_min:x_max, y_min:y_max]
    matrix_2 = fft_1[x_min_2:x_max_2, y_min_2:y_max_2]

    B = 0.5
    b = np.random.uniform(0, B)
    array3 = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)

    A = 5
    a = np.random.uniform(0, A)
    array1 = np.random.uniform(-a, a, size=matrix_1.shape)

    c_down = np.random.uniform(-a, 1 - b)
    c_up = np.random.uniform(1 + b, a)
    array2 = np.random.uniform(c_down, c_up, size=matrix_2.shape)

    fft_1 = fft_1 * array3
    fft_1[x_min_2:x_max_2, y_min_2:y_max_2] = matrix_2 * array2
    fft_1[x_min:x_max, y_min:y_max] = matrix_1 * array1

    if preserve_dc:
        fft_1[dc_index] = original_dc

    img_out = np.fft.ifftn(fft_1)
    new_image = np.clip(img_out, 0, 255).astype(np.uint8)
    return Image.fromarray(new_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='data/cifar/cifar-10-batches-py/data_batch_1')
    parser.add_argument('--idx', type=int, default=8, help='CIFAR image index (8 = a ship in batch 1)')
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--outdir', default='outputs/dc_visual_check')
    parser.add_argument('--panels', default='original_nodc_dc',
                        choices=['original_dc', 'nodc_dc', 'original_nodc_dc'],
                        help='original_dc: original vs dc-fixed. nodc_dc: mrt without dc-fix vs mrt with dc-fix. '
                             'original_nodc_dc: original, mrt without dc-fix, mrt with dc-fix.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_cifar_batch(args.batch)
    orig = row_to_image(data[args.idx])

    dc_fixed = two_region_freqtune(orig, preserve_dc=True, seed=args.seed)
    no_dc = two_region_freqtune(orig, preserve_dc=False, seed=args.seed)

    cell = 32 * args.scale

    if args.panels == 'original_dc':
        panels, out_name = [orig, dc_fixed], 'original_vs_dc_fixed.png'
    elif args.panels == 'nodc_dc':
        panels, out_name = [no_dc, dc_fixed], 'mrt_nodc_vs_dc.png'
    else:
        panels, out_name = [orig, no_dc, dc_fixed], 'original_nodc_dc.png'

    canvas = Image.new('RGB', (cell * len(panels), cell), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel.resize((cell, cell), Image.NEAREST), (cell * i, 0))

    out_path = os.path.join(args.outdir, out_name)
    canvas.save(out_path)
    print('saved', out_path, canvas.size)


if __name__ == '__main__':
    main()
