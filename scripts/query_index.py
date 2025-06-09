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
    p.add_argument("-i", "--input", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument('--db', nargs=1, required=True,
                   help='specify path built index')
    p.add_argument('-k', type=int, nargs=1, default=5,
                   help='number of approx nearest neighbors')
    p.add_argument('--tool', type=str, nargs=1, default='flann',
                   help='Specify tool')
    p.add_argument('--dataset', nargs=1, required=True,
                   help='original dataset used to build index')
    return p.parse_args(argv)


def output_nn(nn, distances, query_label, label):
    assert (len(query_label) == nn.shape[0])
    print(distances.shape)
    s = ''
    for i, query_seq in enumerate(query_label):
        for j, seq_idx in enumerate(nn[i]):
            s += query_seq + '\t' + \
                label[seq_idx] + '\t' + str(distances[i][j]) + '\n'
    print(s)


def flann_query(dataset, query, args):
    flann = FLANN()
    set_distance_type('euclidean')
    flann.load_index(args.db[0], dataset)
    db_idx, distances = flann.nn_index(query, num_neighbors=args.k[0])
    return db_idx, distances


def faiss_query(dataset, query, args):
    index = faiss.read_index(args.db[0])
    distances, db_idx = index.search(query, args.k[0])
    return db_idx, distances


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    query_label, query = utils.read_h5_dataset(args.input[0])
    label, dataset = utils.read_h5_dataset(args.dataset[0])
    if args.tool[0] == 'flann':
        db_idx, distances = flann_query(dataset, query, args)
    elif args.tool[0] == 'faiss':
        db_idx, distances = faiss_query(dataset, query, args)
    print(f'Time to query index: {time.time()-t}')
    output_nn(
        db_idx,
        distances,
        query_label=query_label,
        label=label,
    )
