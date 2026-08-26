import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet


def test(net, test_loader):
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--model', default='wrn', choices=['wrn', 'densenet', 'resnext'])
    parser.add_argument('--eval-batch-size', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        test_data = datasets.CIFAR10(os.path.join(args.data_path, 'cifar'), train=False,
                                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]), download=True)
        num_classes = 10
    else:
        test_data = datasets.CIFAR100(os.path.join(args.data_path, 'cifar'), train=False,
                                      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]), download=True)
        num_classes = 100

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    if args.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif args.model == 'wrn':
        # default layers/widen-factor should match training; use common defaults
        net = WideResNet(40, num_classes, 4, 0.3)
    else:
        net = resnext29(num_classes=num_classes)

    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    ck = torch.load(args.checkpoint, map_location='cpu')
    # find state dict key
    if 'state_dict' in ck:
        state = ck['state_dict']
    elif 'model' in ck:
        state = ck['model']
    else:
        state = ck

    # load properly (handle DataParallel prefixes)
    try:
        net.load_state_dict(state)
    except RuntimeError:
        # strip module.
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        net.load_state_dict(new_state)

    test_loss, test_acc = test(net, test_loader)
    print('Clean Test Loss {:.3f} | Clean Test Error {:.2f}'.format(test_loss, 100 - 100. * test_acc))


if __name__ == '__main__':
    main()
