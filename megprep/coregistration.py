# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step1: Perform automated coregistration using existing algorithms.
Step2: Enhance algorithms for higher precision, formatted reports, and streamlined workflows.
"""
import os
import argparse
import numpy as np
import pandas as pd
import mne
import yaml
from mne.coreg import Coregistration
from pathlib import Path
from utils import start_xvfb,stop_xvfb,set_random_seed,str2bool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# set random seed
set_random_seed(2025)

def perform_coregistration(raw_file_path, subjects_dir, fiducials="estimated", fiducials_file=None,
                           output_dir='.', config=None, visualize=False):
    """
    Perform automated MEG-MRI coregistration, fit fiducials, and ICP registration.

    Parameters:
    - raw_file_path: str or Path, path to the raw MEG data file.
    - subjects_dir: str or Path, directory where subjects' anatomical data is stored.
    - fiducials: str, the type of fiducials to use ("estimated" or "manual").
    - fiducials_file: str or Path, path to a text file containing manual fiducial coordinates (if fiducials="manual").
    - output_dir: str or Path, path to save the transformation and figures.
    - config: dict, configuration parameters for coregistration.
    - visualize: plot coregistration.
    """
    logger.info("Loading raw MEG data...")
    raw = mne.io.read_raw_fif(raw_file_path, verbose=False)
    info = raw.info
    output_dir = Path(output_dir)

    subject = Path(subjects_dir).stem
    fs_subjects_dir = Path(subjects_dir).parent

    logger.info(f"Subject ID: {subject}")
    save_trans_path = output_dir / "coreg-trans.fif"

    if os.path.exists(save_trans_path):
        print(f"The file {save_trans_path} already exists, and the data will not be overwritten.")
    else:

        # Handle fiducials
        if fiducials == "manual" and fiducials_file:
            if not Path(fiducials_file).exists():
                raise FileNotFoundError(f"Fiducials file {fiducials_file} does not exist.")
            fiducials_data = np.loadtxt(fiducials_file)
            fiducials_dict = {
                'nasion': fiducials_data[0],
                'lpa': fiducials_data[1],
                'rpa': fiducials_data[2]
            }
            coreg = Coregistration(info, subject, fs_subjects_dir, fiducials=fiducials_dict)
            logger.info("Using manual fiducials.")
        else:
            coreg = Coregistration(info, subject, fs_subjects_dir, fiducials=fiducials)
            logger.info("Using estimated fiducials.")

        # Plot initial alignment
        plot_kwargs = dict(
            subject=subject,
            subjects_dir=fs_subjects_dir,
            surfaces="head-dense",#,
            dig=True,
            mri_fiducials='estimated',
            meg={"helmet": 0.0, "sensors": 0.0, 'ref': 1},#('helmet', 'sensors', 'ref')
            # show_axes=True,
            # coord_frame="mri",
        )


        plot_flag=visualize
        try:
            display_number = start_xvfb()
            fig = mne.viz.create_3d_figure((10, 10))
            fig.plotter.close()
        except Exception as e:
            logger.error(e)
            plot_flag = False

        white = (1.0, 1.0, 1.0)
        gray = (0.9, 0.9, 0.9)
        black = (0.0, 0.0, 0.0)
        if plot_flag:
            try:
                logger.info(f"Plotting initial alignment...")
                fig = mne.viz.create_3d_figure((400, 400), bgcolor=black)
                mne.viz.plot_alignment(info, fig=fig, trans=coreg.trans,sensor_colors='red',**plot_kwargs)
                fig.plotter.screenshot(output_dir / f"{subject}_coreg_initial.png")
                fig.plotter.close()
            except RuntimeError as e:
                logger.error(e)


        # Fit fiducials
        logger.info("Fitting fiducials...")
        coreg.fit_fiducials(verbose=True)
        if plot_flag:
            try:
                logger.info("Plotting coreg_fiducials...")
                fig = mne.viz.create_3d_figure((400, 400), bgcolor=black)
                mne.viz.plot_alignment(info, fig=fig, trans=coreg.trans, **plot_kwargs)

                fig.plotter.screenshot(output_dir / f"{subject}_coreg_fiducials.png")
                fig.plotter.close()
            except RuntimeError as e:
                logger.error(e)

        # Perform ICP registration
        logger.info("Performing ICP registration...")
        print("config.get('grow_hair', 0):",config.get('grow_hair', 0))
        coreg.set_grow_hair(config.get('grow_hair', 0))
        coreg.omit_head_shape_points(distance=config.get('omit_head_shape_points', 5.0) / 1000)  # Default: 5 mm
        coreg.fit_icp(**config.get('icp'))
        if plot_flag:
            try:
                logger.info("Plotting coreg_icp...")
                fig = mne.viz.create_3d_figure((400, 400), bgcolor=black)
                mne.viz.plot_alignment(info, fig=fig, trans=coreg.trans, **plot_kwargs)
                fig.plotter.screenshot(output_dir / f"{subject}_coreg_icp.png")
                fig.plotter.close()
            except RuntimeError as e:
                logger.error(e)

        # Fine tune registration
        logger.info("Fine tuning ICP registration...")
        try:
            coreg.fit_icp(**config.get('finetune_icp'))
        except ValueError as e:
            logger.error(f"ValueError: Internal algorithm failed to converge.{e}")
            print(coreg.trans)

        if plot_flag:
            try:
                logger.info("Plotting coreg_icp_finetune...")
                fig = mne.viz.create_3d_figure((400, 400), bgcolor=black)
                mne.viz.plot_alignment(info, fig=fig, trans=coreg.trans, **plot_kwargs)
                # Set the 3D view
                # view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
                # mne.viz.set_3d_view(fig, **view_kwargs)

                fig.plotter.screenshot(output_dir / f"{subject}_coreg_icp_finetune.png")
                fig.plotter.close()
            except RuntimeError as e:
                logger.error(e)

        # Compute distances between HSP and MRI
        dists = coreg.compute_dig_mri_distances() * 1e3  # Convert to mm
        logger.info(
            f"Distance between HSP and MRI (mean/min/max): {np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
        )

        # Save distance metrics
        dists_df = pd.DataFrame({
            "dist_min(mm)": [f"{np.min(dists):.2f}"],
            "dist_max(mm)": [f"{np.max(dists):.2f}"],
            "dist_mean(mm)": [f"{np.mean(dists):.2f}"]
        })
        dists_df.to_csv(output_dir / "dists.csv", index=False)

        # Save transformation matrix
        mne.write_trans(save_trans_path, coreg.trans, overwrite=True)
        logger.info(f"Transformation matrix saved to {save_trans_path}")

        stop_xvfb(display_number)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated Coregistration for MEG and MRI data.")

    parser.add_argument('--raw_file', required=True, type=str, help='Path to the raw MEG file')
    parser.add_argument('--subjects_dir', required=True, type=str, help='Path to the subjects directory')
    parser.add_argument('--fiducials', default="estimated", type=str, choices=["estimated", "manual"],
                        help='Type of fiducials to use ("estimated" or "manual")')
    parser.add_argument('--fiducials_file', type=str,
                        help='Path to the file containing manual fiducial coordinates (required if --fiducials is "manual")')
    parser.add_argument('--output_dir', type=str, help='Path to save the transformation and figures')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--visualize', type=str2bool, nargs='?', const=True, default=True, help="Whether to visualize the coregistration (default: True)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Validate output directory
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # debug core
    core_config = """
    omit_head_shape_points: 1 # mm
    grow_hair: 0.0 #mm
    icp:
        n_iterations: 200
        lpa_weight: 1.0
        nasion_weight: 10.0
        rpa_weight: 1.0
        hsp_weight: 10.0
        eeg_weight: 0.0
        hpi_weight: 1.0
    finetune_icp:
        n_iterations: 200
        lpa_weight: 0.0
        nasion_weight: 0.0
        rpa_weight: 0.0
        hsp_weight: 10.0
        eeg_weight: 0.0
        hpi_weight: 0.0
    """
    args.config = core_config
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
    os.environ["DISPLAY"] = ":99" # lp

    # Parse YAML configuration
    config = yaml.safe_load(args.config)

    # Perform the coregistration
    perform_coregistration(
        raw_file_path=args.raw_file,
        subjects_dir=args.subjects_dir,
        fiducials=args.fiducials,
        fiducials_file=args.fiducials_file,
        output_dir=output_dir_path,
        config=config,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
