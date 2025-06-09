#!/usr/bin/env python3
import sys
import argparse
from fasta_reader import read_fasta, write_fasta
import time


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("--query", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument("--target", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    query_id = 'P0AAG5'
    nn = ['P07109', 'P29018', 'P31826', 'P33941', 'Q47538', 'P0A698', 'P77795', 'P75796', 'P75957', 'P33916', 'P36879', 'P32721', 'P0A9R7', 'P16676', 'P37774', 'P77481', 'P33360', 'P68187', 'P16677', 'P0A9X1', 'P07821', 'P77265', 'P10346', 'P10907', 'P0AAF3', 'P0AAF6', 'P0AAH8', 'P30750', 'P0AAH4', 'P16679', 'P0A9S7', 'P0A9U1', 'P75831', 'P43672',
          'P63386', 'P23886', 'P31060', 'P09833', 'P77268', 'P23878', 'P77257', 'P76909', 'P31134', 'P77622', 'P37624', 'P77499', 'P14175', 'P15031', 'P16678', 'P0AAG3', 'P04983', 'P31548', 'P37009', 'P0AAH0', 'P77509', 'P45769', 'P77737', 'P33593', 'P0AAI1', 'P77279', 'P22731', 'P60752', 'P06611', 'P33594', 'P0AAG8', 'Q6BEX0', 'P0A9T8', 'P37388']
    query_fasta = dict(read_fasta(args.query[0]))
    target_fasta = dict(read_fasta(args.target[0]))
    with write_fasta(f"{query_id}.fasta") as file:
        file.write_item(query_id, query_fasta[query_id])
        for seq_id in nn:
            file.write_item(seq_id, target_fasta[seq_id])
    print(f'Time to calculate TSNE: {time.time()-t}')
