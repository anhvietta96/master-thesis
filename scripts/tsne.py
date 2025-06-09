#!/usr/bin/env python3
import sys
import argparse
import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
    p.add_argument("--dataset", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument("--testset", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    return p.parse_args(argv)


def tsne(dataset, testset, datalabel, testlabel, n_pca=50):
    pca = PCA(n_components=n_pca)
    pca_transformed_dataset = pca.fit_transform(dataset)
    pca_transformed_testset = pca.transform(testset)
    dataset_size = dataset.shape[0]
    total_data = np.concatenate(
        (pca_transformed_dataset,
         pca_transformed_testset)
    )
    tsne = TSNE()
    tsne_transformed = tsne.fit_transform(total_data)

    query_id = 'P0AAG5'
    query_idx = testlabel.index(query_id)
    nn = ['P07109', 'P29018', 'P31826', 'P33941', 'Q47538', 'P0A698', 'P77795', 'P75796', 'P75957', 'P33916', 'P36879', 'P32721', 'P0A9R7', 'P16676', 'P37774', 'P77481', 'P33360', 'P68187', 'P16677', 'P0A9X1', 'P07821', 'P77265', 'P10346', 'P10907', 'P0AAF3', 'P0AAF6', 'P0AAH8', 'P30750', 'P0AAH4', 'P16679', 'P0A9S7', 'P0A9U1', 'P75831', 'P43672',
          'P63386', 'P23886', 'P31060', 'P09833', 'P77268', 'P23878', 'P77257', 'P76909', 'P31134', 'P77622', 'P37624', 'P77499', 'P14175', 'P15031', 'P16678', 'P0AAG3', 'P04983', 'P31548', 'P37009', 'P0AAH0', 'P77509', 'P45769', 'P77737', 'P33593', 'P0AAI1', 'P77279', 'P22731', 'P60752', 'P06611', 'P33594', 'P0AAG8', 'Q6BEX0', 'P0A9T8', 'P37388']
    nn_idx = [datalabel.index(n) for n in nn]

    query_tsne = tsne_transformed[dataset_size+query_idx]
    nn_tsne = np.array([tsne_transformed[i] for i in nn_idx])

    ax = plt.subplot()
    ax.scatter(
        tsne_transformed[:, 0],
        tsne_transformed[:, 1],
        label='data'
    )
    ax.scatter(
        [query_tsne[0]],
        [query_tsne[1]],
        label='query_point'
    )
    ax.scatter(
        nn_tsne[:, 0],
        nn_tsne[:, 1],
        label='nn_mmseqs'
    )
    plt.legend()
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    plt.savefig('tsne_1.png')


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    data_label, dataset = utils.read_h5_dataset(args.dataset[0])
    test_label, testset = utils.read_h5_dataset(args.testset[0])
    t = time.time()
    tsne(dataset, testset, data_label, test_label)
    print(f'Time to calculate TSNE: {time.time()-t}')
