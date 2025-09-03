"""
识别MEG数据集的坏道、坏段、Jump等伪迹
运动伪迹识别：眼动、心跳、头动伪迹、哈欠等（有数据集）；

Step1：使用已有的伪迹检测算法
- autoreject
- osl
- mne
- preprep

Step2：基于MEG预训练模型做下游任务的伪迹检测、运动伪迹识别；
"""
import os
import mne
import argparse
import yaml

from osl_ephys.preprocessing.osl_wrappers import detect_badchannels, detect_badsegments
# from tools.osl.osl_wrappers import detect_badchannels, detect_badsegments
from mne.preprocessing import annotate_break,annotate_amplitude,annotate_muscle_zscore
from mne.preprocessing import find_bad_channels_lof
from tools.pyprep.find_noisy_channels import NoisyChannels
from utils import set_random_seed

set_random_seed(2025)

def find_bad_channels(raw,config):
    """Detect bad channels using multiple methods."""
    bad_channels = []

    # PyPrep methods | slow.
    pyprep_config = config.get("pyprep", None)
    if pyprep_config:
        noisy_data = NoisyChannels(raw, random_state=2025)

        if pyprep_config.get('deviation',None):
            noisy_data.find_bad_by_deviation(**pyprep_config['deviation'])
            print("deviation",noisy_data.get_bads())
        if pyprep_config.get('snr', None):
            noisy_data.find_bad_by_SNR()
            print("snr",noisy_data.get_bads())

        if pyprep_config.get('nan_flat', None):
            noisy_data.find_bad_by_nan_flat()
            print("nan_flat",noisy_data.get_bads())

        if pyprep_config.get('hfnoise', None):
            noisy_data.find_bad_by_hfnoise(**pyprep_config['hfnoise'])
            print("hfnoise", noisy_data.get_bads())

        ## very slow,and comment.
        # find bad by ransac
        if pyprep_config.get('ransac', None):
            noisy_data.find_bad_by_ransac(**pyprep_config['ransac'])
            print("ransac", noisy_data.get_bads())

        # find bad by corr
        if pyprep_config.get('correlation', None):
            noisy_data.find_bad_by_correlation(**pyprep_config['correlation'])
            print("correlation", noisy_data.get_bads())

        bad_channels = noisy_data.get_bads()
        bad_channels.extend(bad_channels)
        print("pyprep bad channels: ", bad_channels)


    # PSD method
    if config.get("psd", None):
        std_multiplier = config["psd"].get("std_multiplier",6)
        ch_names = raw.info['ch_names']
        psd = raw.compute_psd().get_data()
        ch_mean_psd = psd.mean(axis=1)
        total_mean, total_std = ch_mean_psd.mean(), ch_mean_psd.std()
        bad_psd_channels = [ch_names[i] for i in range(len(ch_mean_psd)) if ch_mean_psd[i] > (total_mean + std_multiplier * total_std)]
        bad_channels.extend(bad_psd_channels)
        print("psd bad channels:", bad_psd_channels)

    # OSL methods
    osl_config = config.get("osl", None)
    if osl_config:
        _raw = raw.copy()
        _raw.info["bads"] = []
        detect_badchannels(_raw, picks='mag', **osl_config)
        try:
            detect_badchannels(_raw, picks='grad', **osl_config)
        except Exception as e:
            print(e)
        bad_channels.extend(_raw.info["bads"])
        print("osl bad channels:", _raw.info["bads"])

    # MNE methods
    mne_config = config.get("mne", None)
    if mne_config:
        try:
            _raw = raw.copy()
            _raw.info["bads"] = []
            find_bad_channels_lof(_raw, **mne_config.get('find_bad_channels_lof',{}))
            bad_channels.extend(_raw.info["bads"])
            print("mne bad channels:", _raw.info["bads"])
        except Exception as e:
            print(e)
    del _raw
    return bad_channels


def find_bad_segments(raw, config):
    """Detect bad segments using OSL and MNE."""
    annots = raw.annotations
    if config.get("osl",None):
        segment_len = config["osl"].get("segment_len",1000)
        try:
            raw_bad_segments = detect_badsegments(raw, picks='grad', segment_len=segment_len, detect_zeros=True)
        except Exception as e:
            print(e)
            raw_bad_segments = raw

        raw_bad_segments = detect_badsegments(raw_bad_segments, picks='mag', segment_len=segment_len, ref_meg=False, detect_zeros=True)
        annots = raw_bad_segments.annotations + annots

    mne_config = config.get("mne",None)
    if mne_config:
        try:
            if mne_config.get("annotate_muscle_zscore"):
                annot_muscle, scores_muscle = annotate_muscle_zscore(raw,**mne_config.get("annotate_muscle_zscore"))
                annots = annots + annot_muscle
            if mne_config.get("annotate_break"):
                print("debug :::,",mne_config.get("annotate_break"))
                annot_break = mne.preprocessing.annotate_break(raw=raw,**mne_config.get("annotate_break"))
                annots = annots + annot_break
            if mne_config.get("annotate_amplitude"):
                annot_amplitude, _ = annotate_amplitude(raw,**mne_config.get("annotate_amplitude"))
                annots = annots + annot_amplitude
            raw.set_annotations(annots)
        except Exception as e:
            print(e)
    return raw
def main(args):
    print("args.input:", args.input)

    # Parse YAML configuration
    config = yaml.safe_load(args.config)

    base_name = os.path.basename(args.input).split('.')[0]
    output_bad_segments_file = f"{args.output}/{base_name}_bad_segments.txt"
    output_bad_channels_file = f"{args.output}/{base_name}_bad_channels.txt"

    if os.path.exists(output_bad_segments_file) and os.path.exists(output_bad_channels_file):
        print(f"The file {output_bad_segments_file}/{output_bad_channels_file} already exists, and the data will not be overwritten.")
    else:
        # raw = mne.io.read_raw_fif(args.input, preload=True)
        raw = mne.io.read_raw(args.input, preload=True)

        # Detect bad channels
        bad_channels = find_bad_channels(raw,config['find_bad_channels'])
        raw.info['bads'].extend(bad_channels)
        current_bad_channels = set(raw.info['bads'])
        raw.info['bads'] = list(current_bad_channels)
        print("raw.info['bads']:",raw.info['bads'])

        # Detect bad segments
        raw = find_bad_segments(raw,config['find_bad_segments'])

        if not os.path.exists(f"{args.output}"):
            os.makedirs(f"{args.output}")

        # Save results
        raw.annotations.save(output_bad_segments_file, overwrite=True)
        print("raw.annotations[bad segments]:",raw.annotations)

        with open(output_bad_channels_file, 'w') as f:
            for bad_channel in bad_channels:
                f.write(f"{bad_channel}\n")
        try:
            if args.annot and (raw.info['bads'] or raw.annotations):
                print(f"Adding artifact information into {args.input}")
                raw.save(f"{args.input}", overwrite=True)
        except Exception as e:
            print(f"Error overwriting:{args.input}...,\n {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact Detection for MEG Data")
    parser.add_argument("--input", required=True, help="Path to input MEG data file")
    parser.add_argument("--output", required=True, default='.', help="Output directory for results")
    parser.add_argument("--annot", action="store_true", help="Enable annotation saving with MEG raw data")
    parser.add_argument('--config', type=str, default="{}", help='Path to the YAML configuration file')

    args = parser.parse_args()

    # debug
    # args.config =  """
    #     find_bad_channels:
    #         pyprep:
    #             deviation:
    #                 deviation_threshold: 5.0
    #             snr: {}
    #             nan_flat: {}
    #         psd:
    #             std_multiplier: 6
    #         osl:
    #             ref_meg: auto
    #             significance_level: 0.05
    #
    #     find_bad_segments:
    #         osl:
    #             segment_len: 1000 # detect_badsegments
    #         mne:
    #             annotate_muscle_zscore:
    #                 ch_type: mag
    #                 threshold: 12
    #
    # """

    main(args)
