#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.beamformer import apply_lcmv, make_lcmv
import argparse
from utils import handle_yaml_scientific_notation
import yaml
from pathlib import Path

def get_n_rank(subj, sess_id, exclude_ica_file):
    """
    Calculate the n_rank for a given subject and session based on excluded ICA components.
    The ICA components to exclude are read from a text file.
    """
    try:
        subj_name = f"sub-{subj:03d}_ses-{sess_id:03d}"

        # Read excluded ICA components from the file
        with open(exclude_ica_file, 'r') as file:
            excluded_ica = file.readlines()

        # Extract the excluded ICA components for the given subject
        excluded_ica = [comp.strip() for comp in excluded_ica if subj_name in comp]

        len_exclude_ica = len(excluded_ica)
        n_rank = 69 - len_exclude_ica - 1
    except KeyError:
        print(f"ICA components for {subj} session {sess_id} not found, setting n_rank=50.")
        n_rank = 50
    return n_rank


def compute_data_covariance(epochs, cov_min, cov_max, subj_src_path, epoch, n_rank):
    """
    Compute or read the covariance matrix from epochs within a given time window.
    """
    data_cov_fn = os.path.join(subj_src_path, f"{epoch}_tw_{cov_tmin}_{cov_tmax}_epoch-cov.fif")
    data_cov = mne.compute_covariance(epochs, tmin=cov_min, tmax=cov_max, method="auto", rank={'meg': n_rank})
    data_cov.save(data_cov_fn, overwrite=True)


def visualize_covariance_and_spectra(noise_cov, raw_data, subj_src_path):
    """
    Visualize and save the noise covariance matrix and its spectra.
    """
    cov_plot_path = os.path.join(subj_src_path, 'bl_cov.png')
    spectra_plot_path = os.path.join(subj_src_path, 'bl_cov_spectra.png')

    if not os.path.exists(cov_plot_path) or not os.path.exists(spectra_plot_path):
        fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw_data.info)
        fig_cov.savefig(cov_plot_path)
        fig_spectra.savefig(spectra_plot_path)
        plt.close('all')
        print(f"Saved covariance and spectra plots to {subj_src_path}")


def compute_forward_solution(subj_epoch_file, trans, src, bem, fwd_file):
    """
    Compute the forward solution and save it if not already saved.
    """
    if not os.path.exists(fwd_file):
        fwd = mne.make_forward_solution(
            subj_epoch_file, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=None)
        mne.write_forward_solution(fwd_file, fwd, overwrite=True)
    else:
        fwd = mne.read_forward_solution(fwd_file)

    return fwd


def visualize_source_estimate(stc, subject, subjects_dir, subj_src_path, epoch, method, spacing, block):
    """
    Visualize and save the source estimate, showing the peak activation for both hemispheres.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source estimate to visualize (e.g., from dSPM or LCMV).
    subject : str
        The subject identifier.
    subjects_dir : str or Path
        The FreeSurfer subjects directory.
    subj_src_path : str or Path
        The path where the images should be saved.
    epoch : str
        The current epoch identifier.
    method : str
        The method used to generate the source estimate (e.g., 'dSPM' or 'LCMV').
    spacing : str
        The source space resolution (e.g., 'ico5').
    block : bool
        True: interactive visualization.

    Returns
    -------
    None
    """

    for hs in ['lh', 'rh']:  # Loop over left hemisphere and right hemisphere
        # Get the peak vertex and time for the current hemisphere
        vertno_max, time_max = stc.get_peak(hemi=hs)

        # Set up the plotting parameters
        surfer_kwargs = dict(
            subject=subject,
            hemi=hs,
            subjects_dir=subjects_dir,
            clim=dict(kind="percent", pos_lims=[0, 97.5, 100]),  # Color limits
            views="lateral",  # View from the side
            initial_time=time_max,  # Time point of maximum activation
            time_unit="s",
            size=(800, 800),  # Figure size
            smoothing_steps=10,  # Smooth the data for visualization
            brain_kwargs=dict(block=block, show=block)
        )

        # Create the brain visualization object
        brain = stc.plot(**surfer_kwargs)

        # Add the peak activation location as a blue foci on the brain
        brain.add_foci(
            vertno_max,
            coords_as_verts=True,
            hemi=hs,
            color="blue",
            scale_factor=0.6,
            alpha=0.5,
        )

        # Add a title to the figure
        brain.add_text(
            0.1, 0.9, f"{method} (plus location of maximal activation)", "title", font_size=14
        )

        # Save the image to file
        output_file = os.path.join(subj_src_path, f"{epoch}_evoked_{method}-{spacing}-{hs}.png")
        brain.save_image(output_file)
        brain.close()

        print(f"Saved {method} brain plot for {hs} hemisphere to {output_file}")


def compute_dSPM(evoked, fwd, noise_cov, subj_src_path, subject_id, subjects_dir, epoch_label, spacing):
    """
    Compute the dSPM inverse solution and save the results.
    """
    stc_file = os.path.join(subj_src_path, f"{epoch_label}_evoked_dSPM-{spacing}")
    if not os.path.exists(stc_file):
        inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)
        stc = apply_inverse(evoked, inverse_operator, lambda2=1.0 / 9.0, method="dSPM", verbose=True)
        stc.save(stc_file, overwrite=True)
    else:
        stc = mne.read_source_estimate(stc_file)
    visualize_source_estimate(stc, subject_id, subjects_dir, subj_src_path, epoch_label, "dSPM", spacing, block)


def compute_LCMV(evoked, fwd, data_cov, noise_cov, subj_src_path, subject_id, subjects_dir, epoch_label, spacing, n_rank):
    """
    Compute the LCMV beamformer solution and save the results.
    """
    stc_file = os.path.join(subj_src_path, f"{epoch_label}_evoked_lcmv-{spacing}")
    if not os.path.exists(stc_file):
        filters = make_lcmv(evoked.info, fwd, data_cov, noise_cov=noise_cov, pick_ori=None, rank={'meg': n_rank},
                            weight_norm="unit-noise-gain-invariant")
        stc = apply_lcmv(evoked, filters)
        stc.save(stc_file, overwrite=True)
    else:
        stc = mne.read_source_estimate(stc_file)

    visualize_source_estimate(stc, subject_id, subjects_dir, subj_src_path, epoch_label, "LCMV", spacing, block=False)


def process_subject(epoch_file, mri_subject_dir, noise_cov_file, epoch_label, output_dir, config):
    """
    Process a single subject's data for source localization.
    """
    subj = Path(epoch_file).stem
    subject_dir = Path(mri_subject_dir)
    subject_id = subject_dir.stem
    fs_subjects_dir = subject_dir.parent

    try:
        # Get rank based on subject and session
        # n_rank = get_n_rank(subj, sess_id, exclude_ica_file)
        spacing = config.get('spacing')

        # load noise covariance
        noise_cov = mne.read_cov(noise_cov_file)

        # Read epochs and evoked data for "mag"
        epochs = mne.read_epochs(epoch_file)
        evoked = epochs.average().pick(config.get('data_type'))

        # load forward solution
        fwd = mne.load_forward_solution(os.path.join(output_dir, "forward_solution.fif"))

        if 'dspm' in config.get("source_methods").lower():
            # Compute dSPM
            evoked = epochs.average()
            compute_dSPM(evoked, fwd, noise_cov, output_dir, subject_id, fs_subjects_dir, epoch_label, spacing)

        if 'lcmv' in config.get("source_methods").lower():
            n_rank = config.get('LCMV')['n_rank']
            cov_tmin = config.get('LCMV')['cov_tmin']
            cov_tmax = config.get('LCMV')['cov_tmax']

            # Compute data covariance for LCMV
            data_cov = compute_data_covariance(epochs, cov_tmin, cov_tmax, output_dir, epoch_label, n_rank, False)

            # Compute LCMV
            compute_LCMV(evoked, fwd, data_cov, noise_cov, output_dir,subject_id, fs_subjects_dir, epoch_label, spacing, n_rank, f"sub-{subj:03d}")

    except Exception as e:
        print(f"Error processing subject {subj}: {e}")

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process source localization for MEG data.")
    parser.add_argument('--epoch_file', type=str, required=True, help="Path to the epochs file.")
    parser.add_argument('--preproc_dir', type=str, required=True, help="Path to the preprocessed data.")
    parser.add_argument('--mri_subject_dir', type=str, required=True, help="Path to the MRI subject directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Subject id.")
    parser.add_argument('--config', type=str, required=True, help="configure parameters.")
    return parser.parse_args()

def main():
    """
    Main function to run the source localization for a single subject.
    """
    os.environ['DISPLAY'] = ':99'  # set environment for pyvista backend drawing.


    args = parse_arguments()
    handle_yaml_scientific_notation()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # debug
    args.config = """
        source_method:
            - dSPM
            - LCMV

        dSPM：
            spacing: ico4
        LCMV:
            cov_min: 0.01
            cov_max: 0.4
    """
    preproc_dir = args.preproc_dir

    config = yaml.safe_load(args.config)

    process_subject(args.epoch_file, args.mri_subject_dir, args.noise_cov_file, args.output_dir, config)


if __name__ == "__main__":
    main()