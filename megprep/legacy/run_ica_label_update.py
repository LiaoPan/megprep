# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide an inteface to label each ICA component into one of seven categories:
- Brain
- Muscle
- Eye
- Heart
- Line Noise
- Channel Noise
- Other
"""
import os
import argparse
import mne
from pathlib import Path
# from mne_icalabel import label_components

def main():
    args = parse_arguments()
    mne_ic_labels = {'y_pred_proba': [], 'labels': [], 'index': []}

    # Load MEG file
    raw = mne.io.read_raw(args.raw_data_path, preload=True)

    raw_basename = Path(raw.filenames[0]).parent.stem

    # Load the ICA file
    ica = mne.preprocessing.read_ica(args.ica_file)

    # mne-python
    # find which ICs match the EOG pattern
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        print("EOG indices:", eog_indices, eog_scores)
        mne_ic_labels['index'].extend(eog_indices)
        mne_ic_labels['labels'].extend(['EOG']*len(eog_indices))
        mne_ic_labels['y_pred_proba'].extend(eog_scores[eog_indices])
    except RuntimeError as e:
        pass

    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(
        raw, method="correlation", threshold="auto"
    )
    print("ECG indices:", ecg_indices,ecg_scores)
    mne_ic_labels['index'].extend(ecg_indices)
    mne_ic_labels['labels'].extend(['EOG'] * len(ecg_indices))
    mne_ic_labels['y_pred_proba'].extend(ecg_scores[ecg_indices])

    muscle_indices, muscle_scores = ica.find_bads_muscle(raw)
    print("Muscle indices:", muscle_indices,muscle_scores)
    mne_ic_labels['index'].extend(muscle_indices)
    mne_ic_labels['labels'].extend(['EOG'] * len(muscle_indices))
    mne_ic_labels['y_pred_proba'].extend(muscle_scores[muscle_indices])

    print("mne_ic_labels:",mne_ic_labels)

    # ICs_classify
    # ICs_classification.py


    # MEGnet ICA Label
    # Not yet.

    # Label ICA components
    # warning： RuntimeError: Could not find EEG channels in the provided Raw instance.
    # The ICLabel model was fitted on EEG data and is not suited for other types of channels.
    # ic_labels = label_components(raw, ica, method="iclabel")
    # print("[MNE-ICLabel] Component labels:", ic_labels)

    # combine mne-icalabel and mne-python.
    # combined_labels = ic_labels.copy()
    combined_labels = mne_ic_labels

    # Override the IC labels with artifacts detected (EOG, ECG, Muscle)
    # for idx, label in zip(mne_ic_labels['index'], mne_ic_labels['labels']):
    #     combined_labels[idx] = label + combined_labels[idx]

    print("combined_labels:", combined_labels)

    # Save the labelled ICA
    # ica.labels_ = ic_labels
    # ica.save(args.ica_file,overwrite=True)
    output_file = os.path.join(args.output_dir, raw_basename, "ic_labels.txt")
    with open(output_file, "w") as f:
        for idx, label in enumerate(combined_labels):
            f.write(f"Component {idx}: {label}\n")

    # marked artifact IC
    artifact_ic_output_file = os.path.join(args.output_dir, raw_basename, "marked_components.txt")
    labels = combined_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]

    exclude_idx.extend(mne_ic_labels['index'])
    exclude_idx = list(set(exclude_idx))

    with open(artifact_ic_output_file, "w") as f:
        for idx in exclude_idx:
            f.write(f"{idx}\n")

    print(f"Labelled ICA saved to {args.output_dir}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatically label ICA components as artifacts using mne-icalabel.")
    parser.add_argument('--raw_data_path', required=True, help='Path to raw data file')
    parser.add_argument('--ica_file', required=True, help='Path to the precomputed ICA file.')
    parser.add_argument('--output_dir', required=True, help='Path to save the ICA-labelled file.(marked_components.txt)')
    return parser.parse_args()


if __name__ == "__main__":
    main()