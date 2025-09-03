# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified function to read BIDS and raw datasets.
"""
import os
import re
import yaml
import argparse
from pathlib import Path
from tqdm.std import tqdm
from typing import Literal, Optional, List, Union
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report, get_entity_vals

def read_meg_dataset(dataset_dir: Union[str, Path], file_suffix: str = '.fif',
                      dataset_format: Optional[Literal['bids', 'raw','auto']] = None,
                      datatype: Literal['meg'] = 'meg', subjects: Optional[List[str]] = None,
                      sessions: Optional[List[str]] = None, tasks: Optional[List[str]] = None,
                      runs: Optional[List[str]] = None, print_dir: bool = False,
                      bids_report: bool = False) -> List:
    """
    General function to read MEG datasets, supporting both BIDS and raw formats.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the dataset directory.
    file_suffix : str, optional
        File suffix to filter raw dataset files (default is '.fif').
    dataset_format : {'bids', 'raw','auto'}, optional
        Format of the dataset. If None, it will be auto-detected.
    datatype : {'meg'}, optional
        The type of data to read (default is 'meg').
    subjects : list of str, optional
        Specific subjects to load (BIDS format only).
    sessions : list of str, optional
        Specific sessions to load (BIDS format only).
    tasks : list of str, optional
        Specific tasks to load (BIDS format only).
    runs : list of str, optional
        Specific runs to load (BIDS format only).
    print_dir : bool, optional
        If True, prints the directory tree (BIDS format only).
    bids_report : bool, optional
        If True, generates a BIDS report (BIDS format only).

    Returns
    -------
    List
        A list of loaded MEG data objects or file paths.

    Raises
    ------
    ValueError
        If the dataset format is unsupported or the dataset directory is invalid.
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        raise ValueError(f"The specified dataset directory {dataset_dir} is not valid.")

    # Auto-detect dataset format
    if dataset_format == "auto":
        if (dataset_dir / "dataset_description.json").exists():
            dataset_format = 'bids'
        else:
            dataset_format = 'raw'

    # Handle BIDS dataset
    if dataset_format == 'bids':
        if print_dir:
            print_dir_tree(str(dataset_dir), max_depth=3)
        if bids_report:
            print(make_report(str(dataset_dir)))

        bids_path = BIDSPath(root=str(dataset_dir),datatype=datatype)
        entities = bids_path.entities

        for entity in bids_path.entities.keys():
            values = get_entity_vals(str(dataset_dir), entity, with_key=False)
            print("[entity],values:",entity,values)
            if values:
                entities[entity] = values
            else:
                entities[entity] = ['']

        if subjects is not None:
            entities['subject'] = subjects
        if sessions is not None:
            entities['session'] = sessions
        if tasks is not None:
            entities['task'] = tasks
        if runs is not None:
            entities['run'] = runs


        print("entities['session']",entities['session'])
        raw_list = []
        total_iters = len(entities['subject']) * len(entities['session']) * len(entities['task']) * len(entities['run'])
        print("entities['run']",entities['run'])
        print("total_iters", total_iters,len(entities['subject']),len(entities['session']) ,len(entities['task']),len(entities['run']))

        with tqdm(total=total_iters) as pbar:
            for subj in entities['subject']:
                print("debug subject",subj)
                for sess in entities['session']:
                    for tk in entities['task']:
                        if sess == '':
                            sess = None
                        for run in entities['run']:
                            try:
                                if run == '':
                                    bids_path.update(subject=subj, session=sess, task=tk)
                                else:
                                    bids_path.update(subject=subj, session=sess, task=tk, run=run)
                            except (ValueError, RuntimeError) as e:
                                print("BIDS_path Update Error:", e)
                                continue

                            try:
                                # _ = read_raw_bids(bids_path, verbose=False)
                                file_path = bids_path.fpath
                                if os.path.exists(file_path):
                                    print("file_path:", file_path)
                                    raw_list.append(bids_path.copy())
                                else:
                                    print("file_path:",file_path,"does not exist.")
                            except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
                                print("BIDS Parse Error:", e)
                                continue

                            pbar.update(1)

        return raw_list

    # Handle raw dataset
    elif dataset_format == 'raw':
        raw_list = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(file_suffix):
                    raw_list.append(os.path.join(root, file))
        if not raw_list:
            raise ValueError(f"No raw data files found in {dataset_dir}.")

        return raw_list

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}. Supported formats: 'bids', 'raw'.")


def save_raw_list_to_file(raw_list: List[str], output_file: str):
    """
    Save the raw MEG file paths to a text file.

    Parameters
    ----------
    raw_list : List[str]
        List of file paths to be written to the output file.
    output_file : str
        Path to the output text file where the raw list will be saved.
    """
    with open(output_file, 'w') as f:
        for file_path in raw_list:
            f.write(f"{file_path}\n")
    print(f"Saved {len(raw_list)} file paths to {output_file}")

if __name__ == "__main__":
    # Example usage
    # dataset_dir = "/path/to/dataset"
    # /data/liaopan/datasets/MEG-MASC/
    # /data/liaopan/datasets/Holmes_cn/raw
    # # Read BIDS format dataset
    # raw_data_bids = read_meg_dataset(dataset_dir, dataset_format='bids', print_dir=True, bids_report=True)
    # print(f"Loaded {len(raw_data_bids)} MEG datasets from BIDS.")
    #
    # # Read raw format dataset
    # raw_data_raw = read_meg_dataset(dataset_dir, dataset_format='raw')
    # print(f"Loaded {len(raw_data_raw)} MEG datasets from raw format.")

    parser = argparse.ArgumentParser(description="Read MEG datasets in BIDS or raw format.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--dataset_format", type=str, choices=["auto", "bids", "raw"], required=False, help="Format of the dataset.")
    parser.add_argument("--file_suffix", type=str, default=".fif", help="Suffix for raw data files (default: .fif).")
    parser.add_argument("--print_dir", action="store_true", help="Print directory structure (for BIDS format).")
    parser.add_argument("--bids_report", action="store_true", help="Generate BIDS report (for BIDS format).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file to save the raw list of file paths.")
    parser.add_argument('--config', type=str, default="{}", help='YAML configuration parameters')
    args = parser.parse_args()

    # debug
    # args.config = """
    #     # Filter out specific megs
    #     subject_id:
    #         - '01'
    #     session_id:
    #         - '006'
    #     task:
    #         - aef
    # """
    config = yaml.safe_load(args.config)

    raw_list = read_meg_dataset(
        dataset_dir=args.dataset_dir,
        dataset_format=args.dataset_format,
        file_suffix=args.file_suffix,
        print_dir=args.print_dir,
        bids_report=args.bids_report,
        subjects=config.get('subject_id'),
        sessions=config.get('session_id'),
        tasks=config.get('task'),
        runs=config.get('run_id')
    )

    #filtering: keep only the main file, exclude files that are split (e.g. -1.fif, -2.fif, etc.)
    filtered_raw_list = []
    pattern = re.compile(r'-\d+' + re.escape(args.file_suffix) + r'$')

    for file_path in raw_list:
        file_name = os.path.basename(file_path)
        if not pattern.search(file_name):
            filtered_raw_list.append(file_path)
        else:
            print(f"excluded: {file_path}")

    save_raw_list_to_file(filtered_raw_list, args.output_file)
    print(f"Loaded {len(raw_list)} MEG datasets and saved to {args.output_file}.")
