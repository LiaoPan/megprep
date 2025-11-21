# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG Preprocessing.

"""
import multiprocessing
from argparse import ArgumentParser
# from dask.distributed import Client
from osl import preprocessing, utils
from pathlib import Path

# Directories
root_dir = Path("/data/jianing/holmes_cn")
raw_dir = root_dir / "data_cat/raw"
preproc_dir = root_dir / "data_cat/preproc"
Path(preproc_dir).mkdir(parents=True, exist_ok=True)

argp = ArgumentParser()
argp.add_argument('--subj', default='', nargs='+', required=True,
                  help='Set subject')

cli_args = argp.parse_args()

if __name__ == "__main__":
    # Usage:
    ## activate environment: source activate meg_preproc
    ## deactivate enviroment: source deactivate
    # python 2_preprocess_manual_1.py --subj 1 2 3 4 5

    utils.logger.set_up(level="INFO")

    # Setup parallel workers
    ## get num of cores
    # n_workers = multiprocessing.cpu_count()
    # threads_per_worker = 8
    # client = Client(n_workers=int(n_workers/threads_per_worker), threads_per_worker=threads_per_worker)

    config = """
        preproc:
        
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}

    """
    # - set_channel_types: {BIO001: eog, SYS201: ecg}
    # - ica_raw: {picks: meg, n_components: 0.99}
    # - bad_channels: {picks: meg, significance_level: 0.1}
    # - bad_channels: {picks: mag, significance_level: 0.1}
    # - bad_channels: {picks: grad, significance_level: 0.1}

    subjs = [int(i) for i in cli_args.subj]  # # [1,2,3]
    subjs = [12]
    sesssions = [1]  # [1,2,3,4,5,6,7,8,9,10]

    inputs = []

    for sub in subjs:
        for ses in sesssions:
            inputs.append(f"{raw_dir}/sub-{sub:03d}/ses-{ses:03d}/sub-{sub:03d}_ses-{ses:03d}_tsss.fif")

    if inputs != []:
        # Run batch preprocessing
        # ref: https://osl.readthedocs.io/en/latest/autoapi/osl/preprocessing/batch/index.html#osl.preprocessing.batch.run_proc_batch
        preprocessing.run_proc_batch(
            config=config,
            files=inputs,
            outdir=f"{preproc_dir}/",
            overwrite=True,
            dask_client=False,
        )

