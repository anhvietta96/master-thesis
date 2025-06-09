#!/usr/bin/env python3
import sys
import argparse
import time
import csv


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-i", "--input", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument('-r', '--reference', nargs=1, type=str, required=True,
                   help='Specify the reference file')
    return p.parse_args(argv)


def parse_flann(flfile):
    d = {}
    reader = csv.reader(open(flfile, 'r'), delimiter='\t')
    for row in reader:
        if len(row) < 2:
            continue
        d[row[0]] = set(row[1:])
    return d


def nn_accuracy(inputfile, reffile):
    d_input = parse_flann(inputfile)
    d_ref = parse_flann(reffile)
    hit = 0
    count = 0
    for id in d_input.keys():
        assert (id in d_ref.keys() and len(d_input[id]) == len(d_ref[id]))
        hit += len(d_input[id].intersection(d_ref[id]))
        count += len(d_input[id])
    print('Hit: {}'.format(hit))
    print('Count: {}'.format(count))
    print('Accuracy: {:.2f}'.format(hit/count))


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    nn_accuracy(args.input[0], args.reference[0])
    print(f'Time to compare: {time.time()-t}')
