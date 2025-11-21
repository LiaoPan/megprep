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
import logging
import os
import yaml
import argparse
import mne
import pandas as pd
from pathlib import Path
from mne_icalabel import label_components
from tools.ica_classify.ICs_classification import classify_ics
from utils import set_random_seed
from collections import defaultdict

set_random_seed(2025)
# from mne_icalabel import label_components



def calculate_flat_ratio(signal, threshold=0):
    """ check ecg/eog signals and calculate flat ratio
    """
    flat_count = 0  
    total_count = len(signal)  

    for i in range(1, total_count):  
        if abs(signal[i] - signal[i - 1]) <= threshold:  
            flat_count += 1  

    flat_ratio = flat_count / total_count  
    return flat_ratio  


def main():
    args = parse_arguments()

    # debug
    args.config = """
        # detect artifact ICs
        ic_ecg: true
        ic_eog: true
        ic_outlier: true # detect artifact ICs by rules.

        find_bads_eog:
            ch_name: null # or the ch_name of EOG.
            threshold: auto
            l_freq: 1
            h_freq: 10
            start: null
            stop: null
            measure: zscore

        find_bads_ecg:
            ch_name: null # or the ch_name of ECG.
            threshold: auto
            method: ctps
            l_freq: 8
            h_freq: 16
            measure: zscore

        find_bads_muscle:
            threshold: 0.5
            start: null
            stop: null
            l_freq: 7
            h_freq: 45
    
        ica_label: false
        
        ICA_classify:
            meg_vendor: ctf 
            explained_var:
                threshold: 0.1
                ch_type: mag
            find_ecg_ics:
                time_segment: 10 # seconds
                ts_ecg_num_max: 20 # Maximum number of heartbeats expected in the chosen time segment
                l_freq: 0.1
                h_freq: 10
                peak_threshod_coef: 0.4 #Indicates the threshold of the number of ecg signal peak interval (unit: index). (peak_threshod = 0.4 * fs) | # for 1 seconds
                peak_std_threshold_coef: 0.05 #Standard deviation threshold of ecg signal peak interval (unit: index). (peak_std_threshold = peak_std_threshold_coef * fs) | # for 1 seconds
            find_abnormal_psd_ics:
                attention_low_freq: 0
                attention_high_freq: 150
                le_high_freq: 12
                low_freq_energy_threshold: 0.8 # Threshold above which the component is flagged by low-frequency energy ratio
    """

    # Parse YAML configuration
    config = yaml.safe_load(args.config)


    mne_ic_labels = {'y_pred_proba': [], 'labels': [], 'index': []}

    # Load MEG file
    raw = mne.io.read_raw(args.raw_data_path, preload=True)

    raw_basename = Path(raw.filenames[0]).parent.stem

    artifact_ic_output_file = os.path.join(args.output_dir, raw_basename, "marked_components.txt")

    os.makedirs(os.path.dirname(artifact_ic_output_file), exist_ok=True)

    # Check if the file exists
    if os.path.exists(artifact_ic_output_file):
        print(f"The file {artifact_ic_output_file} already exists, and the data will not be overwritten.")
    else:
        ic_ecg = []
        ic_eog = []
        ic_outlier = []
        scores_dict = {'ecg': [], 'ecg_indices': [], 'eog': [], 'eog_indices': []}
        scores_dict = defaultdict(list, scores_dict)
        # Load the ICA file
        ica = mne.preprocessing.read_ica(args.ica_file)

        if config.get('mne_algorithm',True):
            # mne-python
            # find which ICs match the EOG pattern
            try:
                # check eog flat.
                if config["find_bads_eog"]['ch_name'] is not None:
                    for ref_ch_name in config["find_bads_eog"]['ch_name']:
                        logging.info("EOG Ref Channel " + ref_ch_name + "")
                        config["find_bads_eog"]['ch_name'] = ref_ch_name
                        print("EOG Ref Channel: " + ref_ch_name + "")
                        print("Measure Methods:",config["find_bads_eog"]['measure'])
                        print("Reference Channel Name: ",config["find_bads_eog"]['ch_name'])
                        eog_signal = raw.copy().pick(ref_ch_name).get_data()
                        flat_ratio = calculate_flat_ratio(eog_signal)
                        logging.info("The flat ratio of eog signal:",flat_ratio)
                        if flat_ratio > 0.1:
                            config["find_bads_eog"]['ch_name'] = None

                        eog_indices, eog_scores = ica.find_bads_eog(raw,**config.get("find_bads_eog", {}))
                        logging.info("EOG indices:{}, {}".format(eog_indices,eog_scores))
                        print(f"EOG indices({ref_ch_name}):{eog_indices}_eog_scores:{eog_scores}")
                        mne_ic_labels['index'].extend(eog_indices)
                        mne_ic_labels['labels'].extend(['EOG']*len(eog_indices))
                        mne_ic_labels['y_pred_proba'].extend(eog_scores[eog_indices])
                        ic_eog.extend(eog_indices)
                        scores_dict['eog'].extend(eog_scores[eog_indices])
                        scores_dict['eog_indices'].extend(eog_indices)
            except Exception as e:
                logging.error(f"[MNE-Python] Error:{e}")

            ic_eog = list(set(ic_eog))
            print(f"[MNE-Python]ic_eog:{ic_eog}")
            # find which ICs match the ECG pattern
            try:
                # check ecg flat.
                if config["find_bads_ecg"]['ch_name'] is not None:
                    ecg_signal = raw.copy().pick(config["find_bads_ecg"]['ch_name']).get_data()
                    flat_ratio = calculate_flat_ratio(ecg_signal)
                    print("The flat ratio of ecg signal:", flat_ratio)
                    if flat_ratio > 0.1:
                        config["find_bads_ecg"]['ch_name'] = None
                ecg_indices, ecg_scores = ica.find_bads_ecg(raw, **config.get("find_bads_ecg", {}))
                print("ECG indices:", ecg_indices,ecg_scores)
                mne_ic_labels['index'].extend(ecg_indices)
                mne_ic_labels['labels'].extend(['ECG'] * len(ecg_indices))
                mne_ic_labels['y_pred_proba'].extend(ecg_scores[ecg_indices])
                ic_ecg.extend(ecg_indices)
                scores_dict['ecg'] = ecg_scores[ecg_indices]
                scores_dict['ecg_indices'] = ecg_indices
            except Exception as e:
                logging.error(e)

            try:
                # Muscle-related ICs
                muscle_indices, muscle_scores = ica.find_bads_muscle(raw,**config.get("find_bads_muscle", {}))
                print("Muscle indices:", muscle_indices,muscle_scores)
                mne_ic_labels['index'].extend(muscle_indices)
                mne_ic_labels['labels'].extend(['MUSCLE'] * len(muscle_indices))
                mne_ic_labels['y_pred_proba'].extend(muscle_scores[muscle_indices])
                ic_outlier.extend(muscle_indices)
            except RuntimeError as e:
                logging.error(e)

            print("mne_ic_labels:",mne_ic_labels)

        if config.get("rules_algorithm",True):
            print("#"*50,"[ICA_classify]","#"*50)
            # ICA_classify[custom]
            marked_ics = []
            marked_ics_dict = {}
            try:
                ica_fit_file = args.ica_file
                ica_root_dir = Path(args.ica_file).parent
                ica_source_file = ica_root_dir / "ica_sources.fif"
                explained_var_file = ica_root_dir / "ica_explained_var.jl"
                marked_ics,marked_ics_dict = classify_ics(ica_source_file,ica_fit_file,explained_var_file,config.get("ICA_classify",{}))
                # double check.
                ic_ecg.extend(marked_ics_dict['ic_ecg'])
                ic_eog.extend(marked_ics_dict['ic_eog'])
                ic_outlier.extend(marked_ics_dict['ic_outlier'])
                scores_dict['ecg_indices'].extend(ic_ecg)
                scores_dict['eog_indices'].extend(ic_eog)
                scores_dict['ic_outlier'].extend(ic_outlier)

                print("ic_ecg:",ic_ecg)
                print("ic_eog:",ic_eog)
                print("ic_outlier:",ic_outlier)

            except Exception as e:
                logging.error(f"ICA_classify Error:{e}")

        if config.get("ica_label",True):
            # Label ICA components
            # warning： RuntimeError: Could not find EEG channels in the provided Raw instance.
            # The ICLabel model was fitted on EEG data and is not suited for other types of channels.
            # ic_labels = label_components(raw, ica, method="iclabel")
            # print("[MNE-ICLabel] Component labels:", ic_labels)
            ic_labels = label_components(raw, ica, method="megnet")  # slow in cpu.
            _ic_list = []
            for idx, ic_l in enumerate(ic_labels["labels"]):
                if "heart beat" == ic_l and (idx not in scores_dict["ecg_indices"]):
                    scores_dict["ecg"].append(ic_labels["y_pred_proba"][idx])
                    scores_dict["ecg_indices"].append(idx)
                    ic_ecg.append(idx)
                    _ic_list.append(idx)
                elif (("eye blink" == ic_l) or ("eye movement" == ic_l)) and (idx not in scores_dict["eog_indices"]):
                    scores_dict["eog"].append(ic_labels["y_pred_proba"][idx])
                    scores_dict["eog_indices"].append(idx)
                    ic_eog.append(idx)
                    _ic_list.append(idx)
            print("[MNE-ICLabel] Component labels:", _ic_list)


        output_file = os.path.join(args.output_dir, raw_basename, "ecg_eog_scores.json")
        pd.Series(scores_dict).to_json(
            output_file,
            orient="index",
            indent=4,
            force_ascii=False
        )

        # marked artifact IC
        exclude_idx = []
        if config.get('ic_ecg'):
            exclude_idx.extend(ic_ecg)
        if config.get('ic_eog'):
            exclude_idx.extend(ic_eog)
        if config.get('ic_outlier'):
            exclude_idx.extend(ic_outlier)


        # exclude_idx.extend(mne_ic_labels['index'])
        # exclude_idx.extend(marked_ics)
        exclude_idx = sorted(list(set(exclude_idx)))
        print(f"run_ica_label - Exclude ICs:{exclude_idx}")

        with open(artifact_ic_output_file, "w") as f:
            for idx in exclude_idx:
                f.write(f"{idx}\n")

        print(f"Labelled ICA saved to {args.output_dir}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatically label ICA components as artifacts using mne-icalabel.")
    parser.add_argument('--raw_data_path', required=True, help='Path to raw data file')
    parser.add_argument('--ica_file', required=True, help='Path to the precomputed ICA file.')
    parser.add_argument('--output_dir', required=True, help='Path to save the ICA-labelled file.(marked_components.txt)')
    parser.add_argument('--config', type=str, default="{}", help='YAML configuration parameters')
    return parser.parse_args()


if __name__ == "__main__":
    main()