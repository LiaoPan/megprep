#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import mne
import argparse
import sys
from utils import handle_yaml_scientific_notation
import yaml
from pathlib import Path

def compute_forward_solution(subj_epoch_file, trans, src, bem, fwd_file):
    """
    Compute the forward solution and save it if not already saved.

    Parameters
    ----------
    subj_epoch_file : str or Path
        The path to the subject's epoch file.
    trans : str or instance of mne.transforms.Transform
        The MRI to head coordinate transformation.
    src : instance of mne.SourceSpaces
        The source space object.
    bem : instance of mne.ConductorModel
        The BEM solution.
    fwd_file : str or Path
        The name of forward solution.

    Returns
    -------
    fwd : instance of mne.Forward
        The computed forward solution.
    """

    fwd = mne.make_forward_solution(
        subj_epoch_file, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=None
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    return fwd


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Compute and save forward solution for MEG data.")
    parser.add_argument('--epoch_file', type=str, required=True, help="Epoch file")
    parser.add_argument('--trans_file', type=str, required=True, help="trans file")
    parser.add_argument('--epoch_label', type=str, required=True, help="Epoch label (e.g., 'wdonset')")
    parser.add_argument('--src_space_fif', type=str, required=True, help="Source space file path")
    parser.add_argument('--bem_fif', type=str, required=True, help="BEM solution file path")
    parser.add_argument('--output_dir', type=str, required=True, help="Path for output files")
    parser.add_argument('--config', type=str, required=True, help="parameters for forward solution (e.g., 'ico4')")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    handle_yaml_scientific_notation()

    Path(args.output_dir).mkdir(exist_ok=True,parents=True)

    args.config = """
        spacing: ico4
    """

    config = yaml.safe_load(args.config)
    print(config)


    # Define paths to necessary files
    subj_epoch_file = os.path.join()
    trans = mne.read_trans(args.trans_file)

    # Load source space and BEM model
    if not os.path.exists(args.src_space_fif) or not os.path.exists(args.bem_fif):
        print(f"{args.src_space_fif} or {args.bem_fif} is missing, exiting...")
        sys.exit(-1)

    src = mne.read_source_spaces(args.src_fif)
    bem = mne.read_bem_solution(args.bem_fif)

    # Define the forward solution output path
    spacing = config.get('spacing')
    subj_fwd_file = os.path.join(args.output_dir, f"{args.epoch_label}-{spacing}-fwd.fif")

    # Compute forward solution
    compute_forward_solution(args.epoch_file, trans, src, bem, subj_fwd_file)

    print(f"Forward solution saved to: {subj_fwd_file}")


if __name__ == "__main__":
    main()
