from pyflann import *
from tape import ProteinBertModel, TAPETokenizer
import numpy as np
import time
from fasta_reader import read_fasta
import h5py

input_file = '../data/ecoli-per-protein.h5'
test_size = 10


def construct_dataset(ifname):
    with h5py.File(ifname, "r") as file:
        size = len(file.items())
        print(f"Number of entries: {len(file.items())}")
        dataset = np.array(list(file.values())[
                           :(size-test_size)], dtype=np.float32)
        test_set = np.array(list(file.values())[
                            (size-test_size):], dtype=np.float32)
        return dataset, test_set


if __name__ == '__main__':
    dataset, test_set = construct_dataset(input_file)

    t = time.time()
    flann = FLANN()
    set_distance_type('euclidean')
    flann.build_index(dataset, algorithm='kdtree')
    print(f'Time to build index: {time.time()-t}')
    t = time.time()
    print(dataset.shape, test_set.shape)
    nn = flann.nn_index(test_set, num_neighbors=1, checks=5)
    print(f'Time to search index: {time.time()-t}')
    print(nn)
