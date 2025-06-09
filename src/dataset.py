"""
This module contains the dataset class for the fasta dataset.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from fasta_reader import read_fasta
from torch.utils.data import Dataset
from utils import create_encoding_one_hot, create_encoding


class FastaDataset(Dataset):
    def __init__(self, fasta_file, max_length=164, h5file=None):
        self.fasta_dict = dict(read_fasta(fasta_file))
        if h5file:
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
        else:
            self.sequences = create_encoding(self.fasta_dict, max_length)
            outname = Path(fasta_file).stem + '.h5'
            self.export(outname)
            print(f'Created cache at {outname}')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

    def get_dict(self):
        return self.fasta_dict

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
