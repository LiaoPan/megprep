"""
实现对ICA伪迹成分识别，包括眼跳、心跳、头动等运动伪迹
Step1： 基于ICALable、MEGnet算法来实现目前的Baseline
Step2： 基于新研算法，实现对SQUID、OPM的ICA伪迹成分识别；
"""
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import argparse

def run_ica_labeling(raw_data_path, ica_data_path, output_file):
    raw = mne.io.read_raw_fif(raw_data_path, preload=True)
    ica = ICA.load(ica_data_path)

    # MNE-ICALabel
    labels = label_components(ica, raw.info, method='ica', verbose=True)

    with open(output_file, "w") as f:
        for idx, label in enumerate(labels):
            f.write(f"Component {idx}: {label}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label ICA components using mne-icalabel.")
    parser.add_argument('--raw_data_path', required=True, help='Path to raw data file')
    parser.add_argument('--ica_data_path', required=True, help='Path to saved ICA data file')
    parser.add_argument('--output_file', required=True, help='Path to save the ICA component labels')
    args = parser.parse_args()

    run_ica_labeling(args.raw_data_path, args.ica_data_path, args.output_file)


