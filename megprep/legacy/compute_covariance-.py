#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import mne
import argparse
import matplotlib.pyplot as plt
from utils import handle_yaml_scientific_notation
import yaml

def compute_noise_covariance(data_bl, epochs, output_dir, config):
    """
    Compute the noise covariance matrix from baseline data or epochs and save it if not already saved.

    Parameters
    ----------
    data_bl : mne.io.Raw
        The baseline raw data.
    epochs : mne.Epochs
        The epochs data (optional: used if baseline data is not available).
    output_dir : str or Path
        The path where the covariance file should be saved.


    Returns
    -------
    noise_cov : mne.Covariance
        The computed noise covariance matrix.
    """
    noise_cov_fn = os.path.join(output_dir, 'bl-cov.fif')

    noise_cov_tmin = config.get('noise_cov_tmin', None) # If None start at first sample.
    noise_cov_tmax = config.get('noise_cov_tmax', None) # If None end at last sample.
    n_rank = config.get('n_rank', None)
    reject_by_annotation = config.get('reject_by_annotation', True)

    if epochs is not None:
        # Use epochs data for covariance calculation
        noise_cov = mne.compute_covariance(epochs, tmin=noise_cov_tmin, tmax=noise_cov_tmax, rank={'meg': n_rank})
    else:
        # Use baseline data for covariance calculation
        reject = config.get('reject', dict(grad=4000e-13, mag=4e-12))  # T / m (gradiometers); T (magnetometers)
        noise_cov = mne.compute_raw_covariance(data_bl, reject_by_annotation=reject_by_annotation, reject=reject, method="auto",
                                               rank={'meg': n_rank})

    noise_cov.save(noise_cov_fn, overwrite=True)

    return noise_cov

def visualize_covariance_and_spectra(noise_cov, raw_data, output_dir):
    """
    Visualize and save the noise covariance matrix and its spectra.

    Parameters
    ----------
    noise_cov : mne.Covariance
        The noise covariance matrix.
    raw_data : mne.io.Raw
        The raw MEG/EEG data.
    output_dir : str or Path
        The path to save the output figures.

    Returns
    -------
    None
    """
    # Check if the covariance and spectra plots already exist, if not, generate and save them
    cov_plot_path = os.path.join(output_dir, 'bl_cov.png')
    spectra_plot_path = os.path.join(output_dir, 'bl_cov_spectra.png')

    fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw_data.info)
    fig_cov.savefig(cov_plot_path)
    fig_spectra.savefig(spectra_plot_path)
    plt.close('all')
    print(f"Saved covariance and spectra plots to {output_dir}")


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Compute and visualize noise covariance matrix.")
    parser.add_argument('--data_bl', type=str, required=True, help="Path to baseline raw data file (e.g., .fif file)")
    parser.add_argument('--epochs_file', type=str, help="Path to epochs file (optional if baseline data is used)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path where the covariance file should be saved")
    parser.add_argument('--config', type=str, help="")

    return parser.parse_args()


def main():
    """
    Main function to compute and visualize the noise covariance matrix.
    """
    # Parse command-line arguments
    args = parse_arguments()

    loader = handle_yaml_scientific_notation()
    
    # debug 
    args.config = """
        
        # For baseline epochs
        noise_cov_tmin: -0.2 # Start time (in seconds) for covariance calculation window
        noise_cov_tmax: 0.6 # End time (in seconds) for covariance calculation window
        n_rank: null # Rank used for covariance calculation

        # For baseline raw data (continuous)
        reject_by_annotation: true
        picks: meg
        baseline: null
        reject:
            grad: 4000e-13
            mag: 4e-12
    """
    
    config = yaml.safe_load(args.config)
    print(config)

    # Load baseline data
    data_bl = mne.io.read_raw_fif(args.data_bl)

    # Load epochs if provided, otherwise use baseline data
    epochs = None
    if args.epochs_file:
        epochs = mne.read_epochs(args.epochs_file)

    # Step 1: Compute noise covariance
    noise_cov = compute_noise_covariance(data_bl, epochs, args.output_dir, **config)

    # Step 2: Visualize and save covariance and spectra plots
    visualize_covariance_and_spectra(noise_cov, data_bl, args.output_dir, args.do_redo)


if __name__ == "__main__":
    main()
