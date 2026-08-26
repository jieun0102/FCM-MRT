import os
import pickle
import argparse
import numpy as np
from PIL import Image


def load_cifar_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return batch['data']


def cifar_row_to_image(row):
    arr = row.reshape(3, 32, 32).transpose(1, 2, 0)
    return Image.fromarray(arr.astype(np.uint8), mode='RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='data/cifar/cifar-10-batches-py/data_batch_1')
    parser.add_argument('--outdir', default='outputs/mrt_cifar', help='Output directory')
    parser.add_argument('--num', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_cifar_batch(args.batch)
    num = min(args.num, data.shape[0])

    for i in range(num):
        row = data[i]
        img = cifar_row_to_image(row)
        out_path = os.path.join(args.outdir, f'orig_{i:03d}.png')
        img.save(out_path)
        arr = np.array(img)
        print(f'orig saved: {out_path} dtype={arr.dtype} shape={arr.shape} min/max={arr.min()}/{arr.max()}')


if __name__ == '__main__':
    main()
