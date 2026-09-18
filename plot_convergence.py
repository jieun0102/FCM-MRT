"""Plot test-error-vs-epoch curves from one or more cifar.py training_log.csv
files on the same axes, for comparing convergence speed across runs.

Usage:
    python3 plot_convergence.py <label1> <csv1> [<label2> <csv2> ...] [-o out.png]
"""
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # drop the trailing corruption-summary row cifar.py appends (time(s) == 0)
    rows = [r for r in rows if int(r['time(s)']) > 0]
    epochs = [int(r['epoch']) for r in rows]
    errors = [float(r['test_error(%)']) for r in rows]
    return epochs, errors


def main():
    args = sys.argv[1:]
    out_path = 'convergence.png'
    if '-o' in args:
        idx = args.index('-o')
        out_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if len(args) < 2 or len(args) % 2 != 0:
        print('Usage: python3 plot_convergence.py <label1> <csv1> [<label2> <csv2> ...] [-o out.png]')
        sys.exit(1)

    plt.figure(figsize=(9, 6))
    for i in range(0, len(args), 2):
        label, csv_path = args[i], args[i + 1]
        epochs, errors = load(csv_path)
        plt.plot(epochs, errors, label=label, linewidth=1.5)

    plt.xlabel('Epoch')
    plt.ylabel('Test Error (%)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
