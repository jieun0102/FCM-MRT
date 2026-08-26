import os
import pickle
import argparse
import numpy as np
from PIL import Image
from FreqTune_transform import TwoRegionFreqTune


def load_cifar_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    return data


def cifar_row_to_image(row):
    arr = row.reshape(3, 32, 32).transpose(1, 2, 0)
    return Image.fromarray(arr.astype(np.uint8), mode='RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='data/cifar/cifar-10-batches-py/data_batch_1',
                        help='Path to CIFAR batch file')
    parser.add_argument('--outdir', default='outputs/mrt_cifar', help='Output directory')
    parser.add_argument('--num', type=int, default=5, help='Number of images to process')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = load_cifar_batch(args.batch)
    num = min(args.num, data.shape[0])

    transform = TwoRegionFreqTune(probability=1.0)

    for i in range(num):
        img = cifar_row_to_image(data[i])
        out = transform(img)
        out_path = os.path.join(args.outdir, f'mrt_cifar_{i:03d}.png')
        out.save(out_path)
        print(f'saved: {out_path} size={out.size} min/max={np.array(out).min()}/{np.array(out).max()}')


if __name__ == '__main__':
    main()
