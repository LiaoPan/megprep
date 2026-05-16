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
import logging
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from osl_ephys.preprocessing.osl_wrappers import detect_badchannels, detect_badsegments
# from tools.osl.osl_wrappers import detect_badchannels, detect_badsegments
from mne.preprocessing import annotate_break,annotate_amplitude,annotate_muscle_zscore
from mne.preprocessing import find_bad_channels_lof
from tools.pyprep.find_noisy_channels import NoisyChannels
from utils import set_random_seed,plot_snippets

set_random_seed(2025)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _is_bad_annotation(description):
    return "bad" in str(description).lower()


def _annotation_to_sample_bounds(raw, onset, duration):
    try:
        start_sample, stop_sample = raw.time_as_index(
            [float(onset), float(onset) + float(duration)],
            use_rounding=True,
        )
    except Exception:
        sfreq = float(raw.info.get("sfreq", 1.0) or 1.0)
        start_sample = int(round(float(onset) * sfreq))
        stop_sample = int(round((float(onset) + float(duration)) * sfreq))

    start_sample = max(0, min(int(start_sample), raw.n_times))
    stop_sample = max(0, min(int(stop_sample), raw.n_times))
    if stop_sample <= start_sample and duration > 0:
        stop_sample = min(raw.n_times, start_sample + 1)
    return start_sample, stop_sample


def plot_artifact_mask_heatmap(raw, bad_channels, output_path, max_time_bins=2400):
    """Plot a compact mask of bad channels and bad time spans."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude=[])
    if len(picks) == 0:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])
    if len(picks) == 0:
        excluded_types = {"stim", "eog", "ecg", "emg", "misc", "resp", "chpi", "ias", "syst", "exci"}
        picks = np.array(
            [
                idx
                for idx, channel_type in enumerate(raw.get_channel_types())
                if channel_type not in excluded_types
            ],
            dtype=int,
        )
    if len(picks) == 0 or raw.n_times <= 0:
        logger.warning("Skipping artifact mask heatmap because no plottable data channels were found.")
        return

    channel_names = [raw.ch_names[pick] for pick in picks]
    bad_channel_set = set(bad_channels or [])
    bad_channel_rows = np.array([name in bad_channel_set for name in channel_names], dtype=bool)
    n_channels = len(channel_names)
    n_bins = int(min(max_time_bins, max(240, min(raw.n_times, n_channels * 10))))
    n_bins = max(1, n_bins)
    mask = np.zeros((n_channels, n_bins), dtype=np.uint8)

    bad_segment_count = 0
    bad_segment_duration = 0.0
    for annotation in raw.annotations:
        if not _is_bad_annotation(annotation["description"]):
            continue
        start_sample, stop_sample = _annotation_to_sample_bounds(raw, annotation["onset"], annotation["duration"])
        if stop_sample <= start_sample:
            continue
        start_bin = int(np.floor(start_sample / raw.n_times * n_bins))
        stop_bin = int(np.ceil(stop_sample / raw.n_times * n_bins))
        start_bin = max(0, min(start_bin, n_bins - 1))
        stop_bin = max(start_bin + 1, min(stop_bin, n_bins))
        mask[:, start_bin:stop_bin] = np.maximum(mask[:, start_bin:stop_bin], 1)
        bad_segment_count += 1
        bad_segment_duration += float(annotation["duration"])

    if bad_channel_rows.any():
        mask[bad_channel_rows, :] = np.where(mask[bad_channel_rows, :] == 1, 3, 2)

    duration_sec = float(raw.n_times / (raw.info.get("sfreq", 1.0) or 1.0))
    time_scale = 60.0 if duration_sec >= 120 else 1.0
    time_label = "Time (min)" if time_scale == 60.0 else "Time (s)"
    extent = [0, duration_sec / time_scale, -0.5, n_channels - 0.5]
    x_values = np.linspace(0, duration_sec / time_scale, n_bins)

    cmap = ListedColormap(["#f8fafc", "#ef4444", "#2563eb", "#7e22ce"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    figure_height = max(5.0, min(10.5, 3.2 + n_channels * 0.035))
    fig = plt.figure(figsize=(13.5, figure_height), dpi=180)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(24, 4),
        height_ratios=(3, 16),
        hspace=0.06,
        wspace=0.04,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_heatmap = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_side = fig.add_subplot(grid[1, 1], sharey=ax_heatmap)

    segment_fraction = np.isin(mask, [1, 3]).mean(axis=0) * 100.0
    ax_top.fill_between(x_values, segment_fraction, color="#ef4444", alpha=0.20, linewidth=0)
    ax_top.plot(x_values, segment_fraction, color="#dc2626", linewidth=1.2)
    ax_top.set_ylabel("Bad span\ncoverage (%)", fontsize=8, color="#475467")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=8, colors="#667085")
    ax_top.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax_top.spines[["top", "right"]].set_visible(False)

    ax_heatmap.imshow(mask, aspect="auto", interpolation="nearest", origin="lower", cmap=cmap, norm=norm, extent=extent)
    ax_heatmap.set_xlabel(time_label, fontsize=10)
    ax_heatmap.set_ylabel("Channels", fontsize=10)
    max_y_ticks = 18
    if n_channels <= max_y_ticks:
        tick_idx = np.arange(n_channels)
    else:
        tick_idx = np.unique(np.linspace(0, n_channels - 1, max_y_ticks).astype(int))
    ax_heatmap.set_yticks(tick_idx)
    ax_heatmap.set_yticklabels([channel_names[idx] for idx in tick_idx], fontsize=7)
    ax_heatmap.tick_params(axis="x", labelsize=8)
    ax_heatmap.spines[["top", "right"]].set_visible(False)

    channel_segment_fraction = np.isin(mask, [1, 3]).mean(axis=1) * 100.0
    side_colors = np.where(bad_channel_rows, "#2563eb", "#ef4444")
    ax_side.barh(np.arange(n_channels), channel_segment_fraction, color=side_colors, alpha=0.78, height=0.82)
    ax_side.set_xlabel("Bad\nspan (%)", fontsize=8, color="#475467")
    ax_side.tick_params(axis="x", labelsize=8, colors="#667085")
    ax_side.tick_params(axis="y", left=False, labelleft=False)
    ax_side.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_side.spines[["top", "right", "left"]].set_visible(False)

    legend_handles = [
        Patch(facecolor="#f8fafc", edgecolor="#cbd5e1", label="Clean"),
        Patch(facecolor="#ef4444", label="Bad segment"),
        Patch(facecolor="#2563eb", label="Bad channel"),
        Patch(facecolor="#7e22ce", label="Bad channel + segment"),
    ]
    title = "Artifact mask heatmap"
    subtitle = (
        f"{n_channels} channels | {int(bad_channel_rows.sum())} bad channels | "
        f"{bad_segment_count} bad segments | {bad_segment_duration:.1f}s marked bad"
    )
    fig.suptitle(title, x=0.08, y=0.995, ha="left", fontsize=14, fontweight="bold", color="#111827")
    fig.text(0.08, 0.955, subtitle, ha="left", va="top", fontsize=9, color="#667085")
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.patch.set_facecolor("white")
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Artifact mask heatmap saved to {output_path}")

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
            logger.error(e)
        bad_channels.extend(_raw.info["bads"])
        logger.info(f'osl bad channels: {_raw.info["bads"]}')

    # MNE methods
    mne_config = config.get("mne", None)
    if mne_config:
        try:
            _raw = raw.copy()
            _raw.info["bads"] = []
            find_bad_channels_lof(_raw, **mne_config.get('find_bad_channels_lof',{}))
            bad_channels.extend(_raw.info["bads"])
            logger.info(f'mne bad channels: {_raw.info["bads"]}')
        except Exception as e:
            logger.error(e)
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
            logger.error(e)
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
                logger.info(mne_config.get("annotate_break"))
                annot_break = mne.preprocessing.annotate_break(raw=raw,**mne_config.get("annotate_break"))
                annots = annots + annot_break
            if mne_config.get("annotate_amplitude"):
                annot_amplitude, _ = annotate_amplitude(raw,**mne_config.get("annotate_amplitude"))
                annots = annots + annot_amplitude
            raw.set_annotations(annots)
        except Exception as e:
            logger.error(e)
    return raw
    
def main(args):
    logger.info("args.input:", args.input)

    # Parse YAML configuration
    config = yaml.safe_load(args.config)

    base_name = os.path.basename(args.input).split('.')[0]
    output_bad_segments_file = f"{args.output}/{base_name}_bad_segments.txt"
    output_bad_channels_file = f"{args.output}/{base_name}_bad_channels.txt"

    if os.path.exists(output_bad_segments_file) and os.path.exists(output_bad_channels_file):
        logger.info(f"The file {output_bad_segments_file}/{output_bad_channels_file} already exists, and the data will not be overwritten.")
    else:
        # raw = mne.io.read_raw_fif(args.input, preload=True)
        raw = mne.io.read_raw(args.input, preload=True)

        # Detect bad channels
        bad_channels = find_bad_channels(raw,config['find_bad_channels'])
        raw.info['bads'].extend(bad_channels)
        current_bad_channels = set(raw.info['bads'])
        raw.info['bads'] = list(current_bad_channels)
        logger.info(f"raw.info['bads']:{raw.info['bads']}")

        # Detect bad segments
        raw = find_bad_segments(raw,config['find_bad_segments'])

        if not os.path.exists(f"{args.output}"):
            os.makedirs(f"{args.output}")

        # Save results
        raw.annotations.save(output_bad_segments_file, overwrite=True)
        logger.info(f"raw.annotations[bad segments]:{raw.annotations}")

        interpolated_bads = False
        if config.get('interpolate_bads', False) and raw.info['bads']:
            logger.info(f"Interpolating bad channels: {raw.info['bads']}")
            raw.interpolate_bads(reset_bads=True)
            interpolated_bads = True
            logger.info("Bad channels were interpolated and reset in raw.info['bads'].")

        bad_channels = list(raw.info['bads'])
        with open(output_bad_channels_file, 'w') as f:
            for bad_channel in bad_channels:
                f.write(f"{bad_channel}\n")

        try:
            if (args.annot and (raw.info['bads'] or raw.annotations)) or interpolated_bads:
                logger.info(f"Adding artifact information into {args.input}")
                raw.save(f"{args.input}", overwrite=True)
        except Exception as e:
            logger.error(f"Error overwriting:{args.input}...,\n {e}")

        check_imgs_output_dir = Path(output_bad_channels_file).parent / "check_imgs"
        heatmap_img_out = Path(f"{check_imgs_output_dir}/artifact_mask_heatmap.png")
        try:
            logger.info("Generating artifact mask heatmap...")
            plot_artifact_mask_heatmap(
                raw=raw,
                bad_channels=bad_channels,
                output_path=heatmap_img_out,
            )
        except Exception as e:
            logger.error(f"Error generating artifact mask heatmap: {e}")

        # Generate detailed artifacts check images.
        if config.get('artifact_images_enabled',True):
            device_type = config.get('meg_vendor','')
            seg_fname_img_out = Path(f"{check_imgs_output_dir}/waveform/chn.#/seg_$.jpg")
            seg_fname_chn_out = Path(f"{check_imgs_output_dir}/waveform/channels.jl")
            summary_fname_img_out = Path(f"{check_imgs_output_dir}/overview/chn.#/seg_$.jpg")
            summary_fname_chn_out = Path(f"{check_imgs_output_dir}/overview/channels.jl")

            try:
                logger.info("Generating summary anv waveform...")
                # plot segments
                plot_snippets(
                    fname_fif=args.input,
                    fname_bad_chn=output_bad_channels_file,
                    fname_bad_seg=output_bad_segments_file,
                    fname_img_out=seg_fname_img_out,
                    fname_chn_out=seg_fname_chn_out,
                    device_type=device_type,
                    segment_type="segment",
                    n_chans=30,
                    duration=60,
                    n_jobs=-1,
                )

                # plot summary
                plot_snippets(
                    fname_fif=args.input,
                    fname_bad_chn=output_bad_channels_file,
                    fname_bad_seg=output_bad_segments_file,
                    fname_img_out=summary_fname_img_out,
                    fname_chn_out=summary_fname_chn_out,
                    device_type=device_type,
                    segment_type="summary",
                    duration=200,
                    n_jobs=-1,
                )
            except Exception as e:
                logger.error(e)



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
    #     artifact_images_enabled: true
    #     meg_vendor: '' # 'ctf', 'elekta', '4d', 'kit', 'opm', ''
    # """

    main(args)
