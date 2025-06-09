#!/usr/bin/env python3
import sys
import argparse
import h5py
import numpy as np
import time


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-f", "--flann", nargs=1, type=str,
                   required=True,  help="FLANN-like result")
    p.add_argument("-m", "--mmseqs", nargs=1, type=str,
                   required=True,  help="MMseqs2 result")
    p.add_argument('--normed', action='store_true', default=False,
                   help='L2 norm dataset')
    p.add_argument('--ratio', type=float, default=0.9,
                   help='ratio to split the dataset')
    p.add_argument('--output', nargs='+', default='./',
                   help='path to output directory')
    return p.parse_args(argv)


def split_dataset(ifname, ratio, normed=False):
    with h5py.File(ifname, "r") as file:
        size = len(file.items())
        print(size)
        for v in file.values():
            v = np.array(v)
        print(f"Number of entries: {size}")
        sequences = list(file.keys())
        data = np.array(list(file.values()), dtype=np.float32)
        if normed:
            data = normalize(data, norm='l2', axis=1)
        label, test_label, dataset, test_set = train_test_split(
            sequences,
            data,
            test_size=ratio,
            random_state=42
        )
        return label, dataset, test_label, test_set


def save_data(label, data, name):
    assert (len(label) == len(data))
    with h5py.File(name, "w") as file:
        for i, l in enumerate(label):
            dset = file.create_dataset(l, data=data[i])
    print(f'Wrote {len(label)} sequences in {name}')


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    label, dataset, test_label, test_set = split_dataset(
        args.input[0], args.ratio, args.normed)
    save_data(label, dataset, args.output[0]+'data.h5')
    save_data(test_label, test_set, args.output[0]+'test.h5')
    print(f'Time to split data: {time.time()-t}')
