#!/usr/bin/env python3
import sys
import argparse
import h5py
import time
from fasta_reader import read_fasta


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-i", "--input", nargs=1, type=str,
                   required=True,  help="Specify the input h5 file.")
    p.add_argument("-r", "--reference", nargs=1, type=str,
                   required=True,  help="Specify the reference fasta file.")
    p.add_argument('--output', nargs='+', default='./',
                   help='path to output directory')
    return p.parse_args(argv)


def h5_to_fasta(input, reference_file, output):
    l = [[item.defline, item.sequence]
         for item in read_fasta(reference_file)]
    for l1 in l:
        defline = l1[0]
        m = defline.split('|')
        if len(m) > 1:
            assert (len(m) == 3)
            l1[0] = m[1]
    seqs = dict(l)

    s = ''
    with h5py.File(input, "r") as file:
        for key in file.keys():
            defline = key
            if defline not in seqs.keys():
                print(defline)
            assert (defline in seqs.keys())
            s += f'>{defline}\n{seqs[defline]}\n'
    with open(output, 'w') as out:
        out.write(s)


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    h5_to_fasta(
        input=args.input[0],
        reference_file=args.reference[0],
        output=args.output[0]
    )
    print(f'Time to convert: {time.time()-t}')
