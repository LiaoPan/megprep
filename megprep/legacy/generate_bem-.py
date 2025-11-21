# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomy Data Preprocessing.
- 基于python调用Freesurfer的recon-all，以及BEM生成
"""
import os
import mne
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from mne.bem import make_watershed_bem


# 基于mne-python来计算边界元模型BEM
def gen_bem_from_anat(subject_name, fs_root_dir, display=False, ico=4, conductivity=(0.3,), use_mne_c=False):
    """
    Generate BEM model based on Freesurfer Results.

    Parameters
    ----------
    subject_name:str
        - the name of subject
    fs_root_dir:str
        - the directory where the freesurfer results are saved, just like SUBJECTS_DIR environment variable.
    ico:int
        - The surface ico(icosahedron) downsampling to use, e.g. 5=20484, 4=5120, 3=1280.
    conductivity:tuple
        - for single layer:(0.3,); for three layers:(0.3, 0.006, 0.3)
    display:bool
        - If True, display with GUI.
    """
    fs_root_dir = Path(fs_root_dir)
    subj_head = fs_root_dir / subject_name
    bem_path = subj_head / 'bem'
    bem_path.mkdir(parents=True, exist_ok=True)

    fname_bem_surf = bem_path / f'{subject_name}_ico{ico}_watershed_bem.fif'
    fname_bem_sol = bem_path / f'{subject_name}_ico{ico}_watershed_bem-sol.fif'

    make_watershed_bem(subject=subject_name, subjects_dir=fs_root_dir, overwrite=True, volume='T1', atlas=True,
                       gcaatlas=False, show=False, copy=True)

    bem_surf = mne.make_bem_model(subject=subject_name, subjects_dir=fs_root_dir, ico=ico,
                                  conductivity=conductivity)  # for single layer:(0.3,); for three layers:(0.3, 0.006, 0.3)
    bem_sol = mne.make_bem_solution(surfs=bem_surf)

    mne.write_bem_surfaces(fname_bem_surf, bem_surf, overwrite=True)
    mne.write_bem_solution(fname=fname_bem_sol, bem=bem_sol, overwrite=True)

    for surface_name in ['pial', 'white']:
        fname_src_spac = bem_path / f'{subject_name}_ico{ico}_bem_{surface_name}_surface-src.fif'
        fname_vol_src_spac = bem_path / f'{subject_name}_ico{ico}_bem_{surface_name}_volume-src.fif'
        src_spac = mne.setup_source_space(subject=subject_name, subjects_dir=fs_root_dir, spacing=f'ico{ico}',
                                          surface=surface_name, add_dist="patch")
        vol_src_spac = mne.setup_volume_source_space(subject=subject_name, subjects_dir=fs_root_dir,
                                                     surface=fs_root_dir / subject_name / 'bem' / 'inner_skull.surf',
                                                     verbose=True)
        mne.write_source_spaces(fname_src_spac, src_spac, overwrite=True)
        mne.write_source_spaces(fname_vol_src_spac, vol_src_spac, overwrite=True)

    if display:
        plot_bem_kwargs = dict(subject=subject_name, subjects_dir=fs_root_dir, brain_surfaces='white',
                               orientation='coronal', slices=[50, 100, 150, 200])
        mne.viz.plot_bem(**plot_bem_kwargs)  # type: ignore


if __name__ == "__main__":
    # Parse command line arguments using argparse
    parser = argparse.ArgumentParser(description="Generate BEM model based on Freesurfer Results")
    parser.add_argument('--subject_name', type=str, required=True, help='Subject name')
    parser.add_argument('--fs_root_dir', type=str, required=True, help='Freesurfer subjects directory')
    parser.add_argument('--ico', type=int, default=4, help='Icosahedron downsampling (default: 4)')
    parser.add_argument('--conductivity', type=list, nargs='+', default=[0.3], help='Conductivity values (default: 0.3)')
    parser.add_argument('--display', action='store_true', help='If set, display BEM with GUI')

    args = parser.parse_args()

    # Call gen_bem_from_anat with arguments from the command line
    gen_bem_from_anat(args.subject_name, args.fs_root_dir, display=args.display, ico=args.ico, conductivity=tuple(args.conductivity))