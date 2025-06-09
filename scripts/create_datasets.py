#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import sys
import argparse
import h5py
import numpy as np
import time


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-i", "--input", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument('--normed', action='store_true', default=False,
                   help='L2 norm dataset')
    p.add_argument('--ratio', type=float, default=0.9,
                   help='ratio to split the dataset')
    p.add_argument('--output', nargs='+', default='./',
                   help='path to output directory')
    return p.parse_args(argv)

def get_dataset(h5file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, dataset in h5py_dataset_iterator(h5file):
        yield path, dataset


def split_dataset(ifname, ratio, normed=False):
    with h5py.File(ifname, "r") as file:
        size = len(file.items())
        for v in file.values():
            v = np.array(v)
        print(f"Number of entries: {size}")
        sequences = np.array(list(file.keys()))

        l = [ds[1] for ds in get_dataset(file)]

        data = np.array(l,dtype=np.float32)
        nan_seqs = [nanpos[0] for nanpos in np.argwhere(np.isnan(data))]
        nan_seqs = list(dict.fromkeys(nan_seqs).keys())
        if len(nan_seqs) > 0:
            data = np.delete(data,nan_seqs,axis=0)
            sequences = np.delete(sequences,nan_seqs,axis=0)

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
