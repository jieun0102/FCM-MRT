import os
from PIL import Image
from torchvision import datasets
import FreqTune_transform


def save_comparison(output_dir, mode='uniform', strength=0.0, probability=1.0, idx=1, preserve_dc=False):
    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.CIFAR10(root=os.path.join('data', 'cifar'), train=False, download=True)
    x, _ = dataset[idx]

    orig_path = os.path.join(output_dir, f'original_{idx}.png')
    x.save(orig_path)

    transform = FreqTune_transform.TwoRegionFreqTune(
        probability=probability,
        mode=mode,
        strength=strength,
        preserve_dc=preserve_dc,
    )
    transformed = transform(x)
    suffix = '_preserve_dc' if preserve_dc else ''
    transformed_path = os.path.join(output_dir, f'{mode}_strength_{strength}{suffix}_{idx}.png')
    transformed.save(transformed_path)

    orig_img = Image.open(orig_path).convert('RGB')
    trans_img = Image.open(transformed_path).convert('RGB')
    width = orig_img.width + trans_img.width
    height = max(orig_img.height, trans_img.height)
    composite = Image.new('RGB', (width, height), (255, 255, 255))
    composite.paste(orig_img, (0, 0))
    composite.paste(trans_img, (orig_img.width, 0))
    composite_path = os.path.join(output_dir, f'{mode}_check{suffix}_{idx}.png')
    composite.save(composite_path)
    print('Saved:', orig_path)
    print('Saved:', transformed_path)
    print('Saved:', composite_path)


if __name__ == '__main__':
    save_comparison('outputs/identity_compare', mode='uniform', strength=0.0, probability=1.0, idx=1)
