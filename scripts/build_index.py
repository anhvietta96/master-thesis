#!/usr/bin/env python3
from pyflann import *
import sys
import argparse
import time
import faiss
if __package__ is None or __package__ == '':
    # uses current directory visibility
    import utils
else:
    # uses current package visibility
    from . import utils


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-i", "--input", nargs='+', type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument('--algorithm', nargs='+', default='kdtree',
                   help='specify algorithm to index, must be [linear,kdtree,lsh]')
    p.add_argument('--tool', nargs='+', default='flann',
                   help='specify tool to index, must be [faiss, flann]')
    p.add_argument('--output', nargs='+', default='./',
                   help='path to output directory')
    return p.parse_args(argv)


def flann(dataset, args):
    flann = FLANN()
    set_distance_type('euclidean')
    flann.build_index(dataset, algorithm=args.algorithm[0])
    flann.save_index(args.output[0])


def faiss_build(dataset, args):
    d = dataset.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(dataset)
    faiss.write_index(index, args.output[0])


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    _, dataset = utils.read_h5_dataset(args.input[0])
    if args.tool[0] == 'flann':
        flann(dataset, args)
    elif args.tool[0] == 'faiss':
        faiss_build(dataset, args)
    print(f'Time to build index: {time.time()-t}')
