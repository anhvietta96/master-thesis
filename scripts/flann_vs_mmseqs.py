#!/usr/bin/env python3
import sys
import argparse
import time
import csv
import subprocess
from fasta_reader import read_fasta, write_fasta
from pymsaviz import MsaViz
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
    p.add_argument("-f", "--flann", nargs=1, type=str,
                   required=True,  help="Specify the input flann output.")
    p.add_argument("-m", "--mmseqs", nargs=1, type=str,
                   required=True,  help="Specify the input mmseqs output.")
    p.add_argument("--fasta", nargs=1, type=str,
                   required=True,  help="Input fasta sequences")
    p.add_argument("--do-pairwise", action='store_true',
                   required=False,  help="Input fasta sequences")
    p.add_argument("-o", "--output", nargs=1, type=str,
                   required=True,  help="Output dir")
    return p.parse_args(argv)


def parse_flann(flfile):
    d = {}
    dist = {}
    reader = csv.reader(open(flfile, 'r'), delimiter='\t')
    for row in reader:
        if len(row) < 3:
            continue
        if row[0] not in d.keys():
            d[row[0]] = set()
        if row[0] not in dist.keys():
            dist[row[0]] = {}
        d[row[0]].add(row[1])
        dist[row[0]][row[1]] = row[2]
    return d, dist


def parse_mmseqs(mmfile):
    d = {}
    with open(mmfile, 'r') as f:
        for line in f.readlines():
            split = line.split('\t')
            assert (len(split) == 12)
            query_id = split[0]
            ref_id = split[1]
            if query_id not in d.keys():
                d[query_id] = set()
            d[query_id].add(ref_id)
    return d


def create_fasta(query_id, match_ids, fl_mismatch_ids, mm_mismatch_ids, fasta_file, output_dir):
    fasta_seqs = dict(read_fasta(fasta_file))
    for id in match_ids:
        prefix = query_id + '_' + id + '_match'
        dir_prefix = output_dir + prefix + '/' + prefix
        match_fasta = dir_prefix + '.fasta'
        match_algmnt = dir_prefix + '_alignment.fasta'
        match_stats = dir_prefix + '_stats.tsv'
        with write_fasta(match_fasta) as file:
            file.write_item(query_id, fasta_seqs[query_id])
            file.write_item(id, fasta_seqs[id])
        try:
            aln_length, identity, dist = mafft(
                match_fasta, match_algmnt, match_stats)
            yield id, aln_length, identity, dist, 'Both'
        except RuntimeError as err:
            continue

    for id in fl_mismatch_ids:
        prefix = query_id + '_' + id + '_fl'
        dir_prefix = output_dir + prefix + '/' + prefix
        mismatch_fasta = dir_prefix + '.fasta'
        mismatch_algmnt = dir_prefix + '_alignment.fasta'
        mismatch_stats = dir_prefix + '_stats.tsv'
        with write_fasta(mismatch_fasta) as file:
            file.write_item(query_id, fasta_seqs[query_id])
            file.write_item(id, fasta_seqs[id])
        try:
            aln_length, identity, dist = mafft(
                mismatch_fasta, mismatch_algmnt, mismatch_stats)
            yield id, aln_length, identity, dist, 'FLANN'
        except RuntimeError as err:
            continue

    for id in mm_mismatch_ids:
        prefix = query_id + '_' + id + '_mm'
        dir_prefix = output_dir + prefix + '/' + prefix
        mismatch_fasta = dir_prefix + '.fasta'
        mismatch_algmnt = dir_prefix + '_alignment.fasta'
        mismatch_stats = dir_prefix + '_stats.tsv'
        with write_fasta(mismatch_fasta) as file:
            file.write_item(query_id, fasta_seqs[query_id])
            file.write_item(id, fasta_seqs[id])
        try:
            aln_length, identity, dist = mafft(
                mismatch_fasta, mismatch_algmnt, mismatch_stats)
            yield id, aln_length, identity, dist, 'MMseqs2'
        except RuntimeError as err:
            continue


def mafft(fasta_file, alignment_result, stats_result):
    args = [
        'mafft',
        '--amino', '--quiet',
        '--localpair',
        '--maxiterate', '100',
        fasta_file
    ]
    print(' '.join(args))
    try:
        out = str(subprocess.check_output(args), encoding='utf-8')
        with open(alignment_result, 'w') as outfile:
            outfile.write(out)
        '''mv = MsaViz(alignment_result, wrap_length=60, show_count=True)
        mv.savefig(png_result)'''
        statistics, aln_length, identity, dist = utils.alignment_stats(
            alignment_result
        )
        with open(stats_result, 'w') as outfile:
            outfile.write(statistics)
        return aln_length, identity, dist
    except subprocess.CalledProcessError as error:
        print(f'Error: {fasta_file}')
        raise RuntimeError(f'Error: {fasta_file}')


def compare(fl, mm, fldist, fasta_file, output_dir,do_pairwise=False):
    result = 'Query_id\tAccuracy\tPrediction size\tReference size\n'
    pairwise = 'Query_id\tTarget_id\tMatched\tAlignment length\tIdentity\tp-Distance\tFLANN Dist\n'
    accurate = 0
    total = 0
    for query_id, mmset in mm.items():
        if query_id not in fl.keys():
            print(f'{query_id} not found')
            continue
        flset = fl[query_id]
        intersect = flset.intersection(mmset)
        result += f'{query_id}\t{len(intersect)/len(mmset):.2f}\t{len(flset)}\t{len(mmset)}\n'
        accurate += len(intersect)
        total += len(mmset)
        if do_pairwise:
            for target_id, aln_length, identity, dist, matched in create_fasta(
                query_id=query_id,
                match_ids=list(intersect),
                fl_mismatch_ids=list(flset-mmset),
                mm_mismatch_ids=list(mmset-flset),
                fasta_file=fasta_file,
                output_dir=output_dir
            ):
                d = f'{float(fldist[query_id][target_id]):.3f}' if target_id in fldist[query_id].keys(
                ) else ''
                pairwise += f'{query_id}\t{target_id}\t{matched}\t{aln_length}\t{identity:.3f}\t{dist:.3f}\t{d}\n'

    result += f'Overall acaccury: {accurate} correct predictions, {accurate/total}'
    return result, pairwise


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    fl, fldist = parse_flann(args.flann[0])
    mm = parse_mmseqs(args.mmseqs[0])
    fasta_file = args.fasta[0]
    output_dir = args.output[0]
    result, pairwise = compare(
        fl, mm, fldist=fldist, fasta_file=fasta_file, output_dir=output_dir, do_pairwise=args.do_pairwise)
    with open(output_dir+'result.tsv', 'w') as outfile:
        outfile.write(result)
    with open(output_dir+'pairwise.tsv', 'w') as outfile:
        outfile.write(pairwise)

    print(f'Time to compare data: {time.time()-t}')
