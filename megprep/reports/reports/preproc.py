# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import streamlit as st
import mne
import pandas as pd
from reports.utils import in_docker,merge_and_deduplicate_annotations
from pathlib import Path
from datetime import timedelta
mne.viz.set_browser_backend('matplotlib')


# set report root dir.
if in_docker():
    report_root_dir = "/output"
else:
    report_root_dir = st.session_state.get("dataset_report_path")

DATA_DIR = os.path.join(report_root_dir, "preprocessed")

# Streamlit 页面标题
st.title("MEG Preprocessing Results")


# 缓存滤波raw
@st.cache_data
def filter_data(_raw, freqrange):
    raw_filtered = _raw.copy().filter(*freqrange, n_jobs=8)
    raw_filtered.set_annotations(_raw.annotations)
    raw_filtered.info['bads'] = _raw.info['bads']
    return raw_filtered


@st.cache_data
def plot_psd(_raw, freqrange):
    psd_fig = _raw.compute_psd().plot(show=False)
    return psd_fig


# 检查数据目录是否存在
if not os.path.exists(DATA_DIR):
    st.error(f"Data directory '{DATA_DIR}' does not exist. Please create it and add .fif files.")
    DATA_DIR = st.text_input("Please specify a new data directory:", "")

# 列出目录下的 .fif 文件
files = sorted([os.path.join(root, f) for root, dirs, _files in os.walk(DATA_DIR) for f in _files if f.endswith("-raw.fif")])

if not files:
    st.warning("No .fif files found in the data directory. Please add some MEG files.")
else:
    # 文件选择器：允许用户选择一个文件
    selected_file = st.sidebar.selectbox("Select a MEG File:", files)

    if selected_file:

        if "last_selected_file" not in st.session_state:
            st.session_state.last_selected_file = selected_file
        elif st.session_state.last_selected_file != selected_file:
            st.cache_data.clear()
            st.session_state.last_selected_file = selected_file

        # 构造文件路径
        file_path = os.path.join(DATA_DIR, selected_file)

        # 加载 MEG 数据
        origin_raw = mne.io.read_raw_fif(file_path, preload=True)
        with st.expander("Raw Data Info", expanded=False):
            st.write(f"Selected File: {selected_file}")
            st.write("Raw Data Info:", origin_raw)
            st.write("Raw Data Info:", origin_raw.info)

        # 显示原始波形图
        # st.subheader("Raw Waveform")
        # fig_raw = raw.plot(n_channels=10, duration=5, show=False, block=False)
        # st.pyplot(fig_raw)

        # 显示滤波后的波形图
        freqrange = st.sidebar.slider('Band-pass frequency range (Hz)', min_value=0, max_value=300,
                                      value=(1, 40))

        time_duration = st.sidebar.selectbox('Duration (s)', [10, 20, 30, 60], index=2)

        channel_range = st.sidebar.selectbox('Channels', [10, 20, 30, 60, len(origin_raw.ch_names)])

        first_time = origin_raw.first_time
        last_time = origin_raw.last_samp / origin_raw.info['sfreq']
        # timerange = st.sidebar.slider('Time (s)', min_value=0.0, max_value=(last_time-first_time-time_duration),
        #                               value=(0.0,))
        st.sidebar.info(f"Raw | First time: {first_time}, last time: {last_time}")
        start_time = st.sidebar.number_input(f'Starting Time (s) | Time Range: [0, {(last_time-first_time-time_duration):.3f}]',
                                             min_value=0., max_value=(last_time-first_time-time_duration))

        # 初始化当前通道的起始索引（使用Session State 来保持按钮点击后的状态）
        if 'start_channel' not in st.session_state:
            st.session_state.start_channel = 12

        # 滑块：选择起始通道索引
        start_channel = st.sidebar.slider('Starting Channel Index', min_value=0,
                                          max_value=len(origin_raw.ch_names) - channel_range,
                                          value=st.session_state.start_channel, step=1)

        # 根据起始通道索引和通道范围选择要显示的通道
        # selected_channels = raw.ch_names[start_channel:start_channel + channel_range]

        # 更新 session_state 中的起始通道索引
        st.session_state.start_channel = start_channel

        ch_col1, ch_col2 = st.sidebar.columns([1, 1])
        # 向前按钮
        with ch_col1:
            if st.button('Prev'):
                if st.session_state.start_channel > 0:
                    st.session_state.start_channel -= channel_range  # 向前翻页，减去channel_range

        # 向后按钮
        with ch_col2:
            if st.button('Next'):
                if st.session_state.start_channel + channel_range < len(origin_raw.ch_names):
                    st.session_state.start_channel += channel_range  # 向后翻页，增加channel_range


        selected_channels = origin_raw.ch_names[st.session_state.start_channel:st.session_state.start_channel + channel_range]

        raw = origin_raw.filter(*freqrange, n_jobs=8)
        # raw = filter_data(origin_raw, freqrange)

        # 显示功率谱密度 (PSD) 图
        st.subheader("Power Spectral Density (PSD)")
        # psd_fig = raw.compute_psd().plot(show=False)
        psd_fig = plot_psd(raw,freqrange)
        st.pyplot(psd_fig)

        st.subheader(f"Filtered Waveform ({freqrange} Hz)")

        subject_id_dir = Path(selected_file).parent.name
        artifact_dir = Path(selected_file).parent.parent / "artifact_report" / subject_id_dir

        bad_channels_file = artifact_dir / f"{subject_id_dir}_preproc-raw_bad_channels.txt"
        bad_segments_file = artifact_dir / f"{subject_id_dir}_preproc-raw_bad_segments.txt"

        ###############################################################################
        # 1) 检查并加载坏道文件 (xxx_bad_channels.txt)
        ###############################################################################
        if os.path.exists(bad_channels_file):
            # st.write(f"Loading bad channels from: {bad_channels_file}")
            with open(bad_channels_file, 'r') as f:
                bad_channels = [ch.strip() for ch in f.read().splitlines() if ch.strip()]
            raw.info['bads'].extend(origin_raw.info['bads'])
            raw.info["bads"].extend(bad_channels)
        else:
            st.write("No bad channels file found.")

        ###############################################################################
        # 2) 检查并加载坏段文件 (xxx_bad_segments.txt)
        ###############################################################################
        bad_segments = []
        if os.path.exists(bad_segments_file):
            # st.write(f"Loading bad segments from: {bad_segments_file}")
            annotations = mne.read_annotations(bad_segments_file)
            old_annot = origin_raw.annotations
            if old_annot != annotations:
                print(old_annot, annotations)
                print(old_annot != annotations, "old_annot != annotations")
                time_format = "%Y-%m-%d %H:%M:%S.%f"
                new_orig_time = (raw.info['meas_date'] + timedelta(seconds=raw.first_time)).strftime(time_format)
                annotations = merge_and_deduplicate_annotations(annotations, old_annot, orig_time=origin_raw.annotations.orig_time)
                raw.set_annotations(annotations)  # cache后，会丢失Annotation，故重新设置. | 会根据raw的first sample，重新计算offset，导致错误的偏移。
                bad_segments = annotations.onset.tolist()
        else:
            st.write("No bad segments file found.")


        # 状态管理，跟踪当前坏段索引
        if 'current_bad_index' not in st.session_state:
            st.session_state.current_bad_index = 0

        # 显示当前坏段的 onset 时间
        print("bad_segments::",bad_segments)
        if bad_segments:
            st.sidebar.markdown(
                "<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>",
                unsafe_allow_html=True
            )
            st.sidebar.markdown("Artifact View")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button('PrevBadSeg'):
                    if st.session_state.current_bad_index > 0:
                        st.session_state.current_bad_index -= 1

            with col2:
                if st.button('NextBadSeg'):
                    if st.session_state.current_bad_index < len(bad_segments) - 1:
                        st.session_state.current_bad_index += 1

            current_segment_onset = bad_segments[st.session_state.current_bad_index]
            st.sidebar.info(f"Current Bad Segment Onset: {current_segment_onset:.2f} s")
            start_time = current_segment_onset - first_time

        fig_filtered = raw.plot(picks=selected_channels, n_channels=channel_range, duration=time_duration,
                                start=start_time,
                                # show_first_samp=True,
                                show=False, block=False, show_scrollbars=False)
        st.pyplot(fig_filtered)

        with st.expander("Artifacts Info", expanded=False):
            st.write(f"Loading bad channels from: {bad_channels_file}")
            st.write(f"Loading bad segments from: {bad_segments_file}")

        # # 显示功率谱密度 (PSD) 图
        # st.subheader("Power Spectral Density (PSD)")
        # # psd_fig = raw.compute_psd().plot(show=False)
        # psd_fig = plot_psd(raw)
        # st.pyplot(psd_fig)

        if "bad_segments" not in st.session_state:
            st.session_state.bad_segments = raw.annotations

        if "bad_channels" not in st.session_state:
            st.session_state.bad_channels = bad_channels

        if "save_triggered_s" not in st.session_state:
            st.session_state.save_triggered_s = False

        if "save_triggered_c" not in st.session_state:
            st.session_state.save_triggered_c = False

        if "save_triggered_r" not in st.session_state:
            st.session_state.save_triggered_r = False

        with st.form("edit_bad_channels"):

            col1, col2 = st.columns([1, 2.4])
            with col1:
                # 显示坏道检测结果
                # st.subheader("Bad Channels")
                st.write("##### Bad Channels")
                raw.info["bads"] = bad_channels
                # st.write("Bad channels:", raw.info["bads"])
                bad_ch_df = st.data_editor(pd.DataFrame(raw.info['bads'], columns=['Bad channels']), num_rows="dynamic")
                bad_ch_df.dropna(inplace=True)
            with col2:
                # st.subheader("Bad Segments")
                st.write("##### Bad Segments")
                # example：raw.annotations.append(onset=10, duration=5, description="Bad segment")
                bad_segments = raw.annotations
                bad_seg_df = st.data_editor(bad_segments.to_data_frame(time_format=None), num_rows="dynamic")
                bad_seg_df.dropna(inplace=True)
                bad_seg_anat = mne.Annotations(onset=bad_seg_df['onset'].tolist(),
                                               duration=bad_seg_df['duration'].tolist(),
                                               description=bad_seg_df['description'].tolist(),
                                               orig_time=origin_raw.annotations.orig_time)
                st.info(f"onset=Waveform_times + {raw.first_time}")
            st.info("💡 Tip: Click elsewhere after editing to ensure changes are captured")
            save_changes = st.form_submit_button("Save Annotations",use_container_width=True)
            if save_changes:
                st.session_state.save_triggered_s = True
                st.session_state.save_triggered_c = True
                st.session_state.save_triggered_r = True
                st.session_state.bad_segments = bad_seg_anat
                st.session_state.bad_channels = bad_ch_df

                bad_seg_anat.save(bad_segments_file, overwrite=True)
                bad_ch_df.to_csv(bad_channels_file, header=False, index=False, sep='\n')
                st.success(f"Saved:{bad_channels_file}")
                st.success(f"Saved:{bad_segments_file}")

                # 修改对应脑磁文件的Annotation和bads info.
                origin_raw.set_annotations(bad_seg_anat)
                print("bad_segments:",bad_seg_anat.to_data_frame(None))
                bad_channels_list = bad_ch_df['Bad channels'].tolist()
                print("bad_ch_df:",bad_channels_list)

                origin_raw.info['bads'] = bad_channels_list
                origin_raw.save(file_path, overwrite=True)
                st.success(f"Overwriting:{file_path}")

                # reset bad seg/channel index view.
                st.session_state.current_bad_index = 0

                st.rerun()
            else:
                st.session_state.save_triggered_s = False
                st.session_state.save_triggered_c = False
                st.session_state.save_triggered_r = False

        # 根据保存状态提供提示
        if st.session_state.save_triggered_c and st.session_state.save_triggered_s and st.session_state.save_triggered_r:
            st.info("Save operation completed.")
        else:
            st.info("No save operation performed.")
