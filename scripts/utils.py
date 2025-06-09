import h5py
import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import math
from collections import Counter


def shannon_entropy(column):
    freqs = Counter(column)
    total = sum(freqs.values())
    entropy = -sum((count/total) * math.log2(count/total)
                   for count in freqs.values() if count > 0)
    return entropy


def percent_identity(seq1, seq2):
    matches = sum(a == b for a, b in zip(seq1, seq2) if a != '-' and b != '-')
    length = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
    return (matches / length) * 100 if length else 0


def p_distance(seq1, seq2):
    mismatches = sum(a != b for a, b in zip(
        seq1, seq2) if a != '-' and b != '-')
    length = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
    return (mismatches / length) if length else 0


def calculate_statistics(alignment):
    aln_length = alignment.get_alignment_length()
    identity = percent_identity(alignment[0].seq, alignment[1].seq)
    dist = p_distance(alignment[0].seq, alignment[1].seq)
    outstring = f"Alignment length: {aln_length}.\n"
    outstring += f"Identity: {identity:.3f}\n"
    outstring += f"p-Distance: {dist:.3f}"

    '''
    # Shannon Entropy per column
    outstring += "== Shannon Entropy per Column ==\n"
    entropies = []
    for col in range(aln_length):
        column = alignment[:, col]
        ent = shannon_entropy(column.replace("-", ""))  # Ignore gaps
        entropies.append(str(ent))
    outstring += "\t".join(entropies) + '\n'

    # Consensus Sequence & Match Rate
    outstring += "\n== Consensus Sequence & Match Rate =="
    consensus = ''
    matches = []
    for col in range(aln_length):
        column = alignment[:, col]
        counter = Counter(column)
        most_common = counter.most_common(1)[0][0]
        consensus += most_common
        match_count = sum(1 for res in column if res == most_common)
        matches.append(str(match_count / n))
    outstring += "Consensus:" + consensus
    outstring += "Match Rates:" + '\t'.join(matches)

    # Gap Frequency per Column
    outstring += "\n== Gap Frequency per Column =="
    gap_freqs = []
    for col in range(aln_length):
        column = alignment[:, col]
        gaps = column.count('-')
        gap_freqs.append(str(gaps / n))
    outstring += '\t'.join(gap_freqs)'''
    return outstring, aln_length, identity, dist


def alignment_stats(alignment_file):
    alignment = AlignIO.read(alignment_file, "fasta")
    return calculate_statistics(alignment)


def read_h5_dataset(ifname):
    with h5py.File(ifname, "r") as file:
        print(f"Number of entries: {len(file.items())}")
        dataset = np.array(list(file.values()), dtype=np.float32)
        label = list(file.keys())
        return label, dataset
