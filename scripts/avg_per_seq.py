import torch
from tape import ProteinBertModel, TAPETokenizer
import numpy as np
import time
from fasta_reader import read_fasta

input_file = '../data/uniprot_sprot.fasta'

model = ProteinBertModel.from_pretrained('bert-base')
# iupac is the vocab for TAPE models, use unirep for the UniRep model
tokenizer = TAPETokenizer(vocab='unirep')


def encode_sequence(s):
    sequence = np.array(list(s))
    t = time.time()
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]
    return sequence_output[0, :].mean(dim=0).detach().numpy(), time.time()-t


encoded_sequences = np.zeros((100, 768))
time_measured = []
for i, item in enumerate(list(read_fasta(input_file))[:100]):
    encoded, elapsed = encode_sequence(item.sequence)
    encoded_sequences[i, :] = encoded
    time_measured.append(elapsed)

print(sum(time_measured)/len(time_measured))
