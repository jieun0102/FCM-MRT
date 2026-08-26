import os
import argparse
import pickle
import numpy as np
from PIL import Image
from FreqTune_transform import TwoRegionFreqTune
import csv


def load_cifar_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return batch['data']


def row_to_image(row):
    arr = row.reshape(3, 32, 32).transpose(1, 2, 0)
    return arr


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


def save_side_by_side(orig, transformed, out_path):
    # orig/transformed are uint8 arrays HxWx3
    o = Image.fromarray(orig)
    t = Image.fromarray(transformed)
    combined = Image.new('RGB', (orig.shape[1] * 2, orig.shape[0]))
    combined.paste(o, (0, 0))
    combined.paste(t, (orig.shape[1], 0))
    combined.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='data/cifar/cifar-10-batches-py/data_batch_1')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--outdir', default='outputs/fcm_mrt_check')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_cifar_batch(args.batch)
    num = min(args.num, data.shape[0])

    transform = TwoRegionFreqTune(probability=1.0)

    csv_path = os.path.join(args.outdir, 'summary.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'psnr', 'orig_min', 'orig_max', 'tr_min', 'tr_max', 'pct_clipped', 'pct_diff_gt5'])

        psnrs = []
        clipped_counts = []
        diffs = []

        for i in range(num):
            orig = row_to_image(data[i])
            orig_u8 = orig.astype(np.uint8)
            pil = Image.fromarray(orig_u8)
            tr_pil = transform(pil)
            tr = np.array(tr_pil).astype(np.uint8)

            # basic checks
            if not np.isfinite(tr).all():
                print(f'Non-finite values at index {i}')

            # stats
            o_min, o_max = int(orig_u8.min()), int(orig_u8.max())
            t_min, t_max = int(tr.min()), int(tr.max())

            # clipping: pixels equal to 0 or 255
            clipped = np.logical_or(tr == 0, tr == 255)
            pct_clipped = 100.0 * clipped.sum() / tr.size

            # percent of pixels that changed by more than 5 (per-channel)
            diff = np.abs(tr.astype(np.int16) - orig_u8.astype(np.int16))
            pct_diff_gt5 = 100.0 * (diff > 5).sum() / diff.size

            val_psnr = psnr(orig_u8, tr)
            psnrs.append(val_psnr)
            clipped_counts.append(pct_clipped)
            diffs.append(pct_diff_gt5)

            writer.writerow([i, val_psnr, o_min, o_max, t_min, t_max, pct_clipped, pct_diff_gt5])

            # save a few sample side-by-side images
            if i < 10:
                out_path = os.path.join(args.outdir, f'cmp_{i:03d}.png')
                save_side_by_side(orig_u8, tr, out_path)

    # summary
    print('summary:')
    print('mean PSNR:', np.mean([p for p in psnrs if np.isfinite(p)]))
    print('median PSNR:', np.median([p for p in psnrs if np.isfinite(p)]))
    print('mean pct_clipped:', np.mean(clipped_counts))
    print('mean pct_diff_gt5:', np.mean(diffs))
    print('results saved to', args.outdir)


if __name__ == '__main__':
    main()
