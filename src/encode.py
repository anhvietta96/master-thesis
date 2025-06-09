"""
Generate sequences with a trained VAE
"""

import os
import torch
import argparse as ap
import h5py
import numpy as np
import time
from torch.utils.data import DataLoader
from fasta_reader import read_fasta, write_fasta
from vae_model import VAE, AE, CAE, CVAE
from constants import max_length, input_dim, latent_dim, hidden_dim
from utils import decode_one_hot_sequence, PAD_TOKEN, create_encoding
from dataset import FastaDataset


def parse_args():
    parser = ap.ArgumentParser(description="Generate sequences with a trained VAE.")
    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help="Path to the trained VAE weights."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Input fasta"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="generated_sequences.fasta",
        help="Path to the output file."
    )
    return parser.parse_args()


def main(
        state_path: str,
        input_file: str,
        output_file: str
):
    '''vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).cuda()
    vae.load_state_dict(torch.load(weights_path))
    vae.eval()'''
    cae = CVAE(latent_dim=latent_dim).cuda()
    cae.load_state_dict(torch.load(state_path)['model_state'])
    cae.eval()
    t = time.time()
    dataset = FastaDataset(input_file, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Time to create encoding: {time.time()-t}s")
    num_seq = len(dataset)
    embeddings = np.empty((num_seq,latent_dim),dtype=np.float32)
    total_time = 0
    i = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.cuda()
            #to_tensor = torch.tensor(input_encoding[i,:,:], dtype=torch.float32).cuda().view((input_dim))
            t = time.time()
            #mean, _ = vae.encode(to_tensor)
            mean, logvar = cae.encode(batch)
            total_time += time.time() - t
            offset = batch.size(0)
            embd = mean.cpu().squeeze().numpy()
            embeddings[i:i+offset,:] = embd
            i += offset

    assert(i == num_seq)
    fasta_dict = dataset.get_dict()
    with h5py.File(output_file,"w") as file:
        for i, seq_id in enumerate(fasta_dict.keys()):
            split = seq_id.split('|')
            if len(split) > 1:
                extracted_id = split[1]
            else:
                extracted_id = seq_id
            file.create_dataset(extracted_id,data=embeddings[i],shape=embeddings.shape[1],dtype=np.float32)

    print(f"Generated sequences saved to {output_file} in {total_time}s")

if __name__ == '__main__':
    args = parse_args()
    state_path = args.state
    inputfile = args.input
    print(inputfile)
    output_file = args.output_file
    main(state_path, inputfile, output_file)
