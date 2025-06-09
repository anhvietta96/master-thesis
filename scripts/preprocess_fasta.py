#!/usr/bin/env python3
import sys
import argparse
import h5py
import numpy as np
import time
from fasta_reader import read_fasta, write_fasta
import matplotlib.pyplot as plt

AMINO_ACIDS = ['A','R','N','D','C','Q','G','H','I','L','K','M','F','P','E','S','T','W','Y','V']

def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-i", "--input", nargs=1, type=str,
                   required=True,  help="Specify the input file.")
    p.add_argument("-m", "--min-length", nargs=1, type=int,
                   required=True,  help="Specify the input file.")
    p.add_argument("-l", "--max-length", nargs=1, type=int,
                   required=True,  help="Specify the input file.")
    p.add_argument("-n", "--num-seq", nargs=1, type=int,
                   required=True,  help="Max number of sequences")
    p.add_argument('--output', nargs='+', default='./',
                   help='path to output directory')
    return p.parse_args(argv)


def histogram(data,max_length):
    data = {k: v for k,v in data.items() if len(v) < max_length}
    seq_lens = [len(seq) for seq in data.values()]
    n, bins, patches = plt.hist(seq_lens,bins=20)
    print(n,bins)
    plt.savefig('seqlen_hist.png')
    return data

def clean_sequence(data):
    cleaned_data = {}
    for seq_id, seq in data.items():
        seq_as_list = [c if c in AMINO_ACIDS else 'X' for c in seq]
        cleaned_data[seq_id] = ''.join(seq_as_list)
    return cleaned_data

def save_fasta(data,output):
    with write_fasta(output) as file:
        for seq_id,seq in data.items():
            file.write_item(seq_id,seq)

def read_dict(inputfile,num_seq,min_length,max_length):
    d = {}
    with open(inputfile,'r') as file:
        linecount = 0
        seq_id = ''
        seq = ''
        for line in file:
            if not line.startswith('>'):
                seq += line[:-1]
            elif linecount < num_seq:
                if len(seq) >= min_length and len(seq) <= max_length:
                    d[seq_id] = seq
                    linecount += 1
                seq = ''
                seq_id = line[1:-1]
            else:
                break
    return d

if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    #data = dict(read_fasta(args.input[0]))
    #data = {k:v for i,(k,v) in enumerate(data.items()) if i < args.num_seq}
    #print(args.num_seq[0])
    data = read_dict(args.input[0],args.num_seq[0],args.min_length[0],args.max_length[0])
    data = histogram(data,args.max_length[0])
    data = clean_sequence(data)
    print(len(data))
    save_fasta(data,args.output[0])
    print(f'Time to split data: {time.time()-t}')

