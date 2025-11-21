# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1： 基于已有算法的自动配准
基于osl实现自动配准
基于mne-python，实现自动配准

Step2： 综合改进已有算法，实现更高精度的自动化配准、格式化自动配准报告；
"""
import os
import argparse
import numpy as np
import pandas as pd
import mne
import yaml
from mne.coreg import Coregistration
from mne.io import read_info
from pathlib import Path


def perform_coregistration(raw_file_path, subjects_dir, fiducials="estimated", fiducials_file=None,
                           output_dir=None,config):
    """
    Perform automated MEG-MRI coregistration, fit fiducials, and ICP registration.

    Parameters:
    - raw_file_path: str or Path, path to the raw MEG data file.
    - subjects_dir: str or Path, directory where subjects' anatomical data is stored.
    - fiducials: str, the type of fiducials to use ("estimated" or "manual").
    - fiducials_file: str or Path, path to a text file containing manual fiducial coordinates (if fiducials="manual").
    - output_dir: str or Path, path to save the transformation and figure.
    """
    # Load MEG data info
    raw = mne.io.read_raw_fif(raw_file_path)
    info = raw.info

    subject = Path(subjects_dir).stem

    # If fiducials is "manual", load the coordinates from the file
    if fiducials == "manual" and fiducials_file:
        fiducials_data = np.loadtxt(fiducials_file)
        fiducials_dict = {
            'nasion': fiducials_data[0],
            'lpa': fiducials_data[1],
            'rpa': fiducials_data[2]
        }
        coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials_dict)
    else:
        # Use "estimated" fiducials
        coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)

    # Initial plot alignment
    plot_kwargs = dict(
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces="head-dense",
        dig=True,
        eeg=[],
        meg="sensors",
        show_axes=True,
        coord_frame="meg",
    )
    fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

    # Save figure if required
    fig.savefig(output_dir / f'{subject}_coreg_initial.png')

    # Fit fiducials
    coreg.fit_fiducials(verbose=True)
    fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

    # Save figure if required
    fig.savefig(output_dir / f'{subject}_coreg_fiducials.png')

    # Perform ICP (Iterative Closest Point) registration
    # Optionally omit head shape points (in meters)
    coreg.omit_head_shape_points(distance=config['omit_head_shape_points'] / 1000)  # 5 mm
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

    # Save figure if required
    fig.savefig(output_dir / f'{subject}_coreg_icp.png')

    # Set the 3D view
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
    mne.viz.set_3d_view(fig, **view_kwargs)

    # Compute distances between HSP and MRI (in mm)
    dists = coreg.compute_dig_mri_distances() * 1e3  # Convert to mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )

    dists_df = pd.DataFrame({"dist_min(mm)": f"{np.min(dists):.2f}",
                          "dist_max(mm)": f"{np.max(dists):.2f}",
                         "dist_mean(mm)": f"{np.mean(dists):.2f}"})
    dists_df.to_csv(os.path.join(output_dir, "dists.csv"), index=False)

    save_trans_path = os.path.join(output_dir,"coreg-trans.fif")
    # Save the transformation matrix
    mne.write_trans(save_trans_path, coreg.trans)
    print(f"Transformation matrix saved to {save_trans_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated Coregistration for MEG and MRI data.")

    parser.add_argument('--raw_file', required=True, type=str, help='Path to the raw MEG file')
    parser.add_argument('--subject', required=True, type=str, help='Subject ID')
    parser.add_argument('--subjects_dir', required=True, type=str, help='Path to the subjects directory')
    parser.add_argument('--fiducials', default="estimated", type=str, choices=["estimated", "manual"],
                        help='Type of fiducials to use ("estimated" or "manual")')
    parser.add_argument('--fiducials_file', type=str,
                        help='Path to the file containing manual fiducial coordinates (required if --fiducials is "manual")')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to save the transformation matrix')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Ensure save paths are valid directories
    output_dir_path = Path(args.output_dir)
    output_dir_path.parent.mkdir(parents=True, exist_ok=True)

    # debug core
    core_config = """
        n_iterations: 20
        lpa_weight: 1.0
        nasion_weight: 10.0
        rpa_weight: 1.0
        hsp_weight: 1.0
        eeg_weight: 1.0
        hpi_weight: 1.0
    """

    # Parse YAML configuration
    config = yaml.safe_load(args.config)


    # Perform the coregistration
    perform_coregistration(
        raw_file_path=args.raw_file,
        subjects_dir=args.subjects_dir,
        fiducials=args.fiducials,
        save_trans_path=output_dir_path,
        config=config,
        fiducials_file=args.fiducials_file,
    )

if __name__ == "__main__":
    main()
