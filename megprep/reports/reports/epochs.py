# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import mne
import streamlit as st
import pandas as pd
import ast  # 用于解析文本中的列表
from pathlib import Path
from reports.utils import in_docker
mne.viz.set_browser_backend('matplotlib')
# 设置页面标题
st.title("MEG Epochs")

# 文件路径
# set report root dir.
if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

DEFAULT_EPOCHS_DIR = report_root_dir / "preprocessed" / "epochs"

epochs_base_path = Path(st.sidebar.text_input(
    "Epochs Report Directory:",
    value=DEFAULT_EPOCHS_DIR
))


subjects_epochs_dirs = sorted([f for f in os.listdir(epochs_base_path)])

if not subjects_epochs_dirs:
    st.warning("No epochs found in the data directory. Please change the data directory.")
else:
    selected_dir = st.sidebar.selectbox("Select a directory:", subjects_epochs_dirs) #subject_id_dir
    print("selected directory: ", selected_dir)

    reject_log_file = epochs_base_path / selected_dir / f"{selected_dir}_preproc-raw_clean_raw_reject_epoch_log.txt"
    epoch_file = epochs_base_path / selected_dir / f"{selected_dir}_preproc-raw_clean_raw-epo.fif"
    epochs = mne.read_epochs(epoch_file, preload=True)

    st.subheader("Epochs Visualization")
    fig_epochs = epochs.plot(show_scrollbars=False)
    st.pyplot(fig_epochs)

    # Plot Event Related Potential / Fields image
    st.subheader("Epochs ERF Visualization")
    fig_erf = epochs.plot_image(picks='meg', combine="mean")
    st.pyplot(fig_erf[0])
    st.pyplot(fig_erf[1])

    st.subheader("Evoked Visualization")
    fig_evoked = epochs.average().plot()
    st.pyplot(fig_evoked)


    # PNG 文件路径
    png_files = [
        f"{selected_dir}_preproc-raw_clean_raw_epoch_onset_psd.png",
        f"{selected_dir}_preproc-raw_clean_raw_epoch_onset_sensors_2d.png",
        f"{selected_dir}_preproc-raw_clean_raw_epoch_onset_sensors_3d.png",
        f"{selected_dir}_preproc-raw_clean_raw_epoch_onset_topo_mag.png",
    ]


    # 展示 PNG 图片
    st.subheader("Visualization Epochs")
    for png_file in png_files:
        png_path = epochs_base_path / selected_dir / png_file
        if png_path.exists():
            st.image(str(png_path), caption=png_file, use_container_width=True)
        else:
            st.warning(f"File not found: {png_file}")

    # 读取 rejected epochs
    print("reject_log_file,",reject_log_file)
    if reject_log_file.exists():
        with open(reject_log_file, "r") as f:
            log_content = f.readlines()

        try:
            rejected_epochs = ast.literal_eval(log_content[0].strip())
            num_epochs = int(log_content[1].split(":")[-1].strip())
            if isinstance(rejected_epochs, list) and rejected_epochs:
                # 展示原始内容
                st.subheader("Epochs Reject Log")

                # 转换为 Pandas DataFrame
                df = pd.DataFrame(rejected_epochs, columns=["Rejected Epochs"])

                # 分页显示
                page_size = st.sidebar.slider("Set Page Size:", min_value=10, max_value=50, value=20)
                page_number = st.number_input(
                    "Select Page Number:",
                    min_value=1,
                    max_value=(len(df) // page_size) + 1,
                    value=1,
                    step=1,
                )

                start_index = (page_number - 1) * page_size
                end_index = start_index + page_size
                st.dataframe(df.iloc[start_index:end_index])

                # 数据分析
                st.subheader("Statistics and Insights")
                st.write(f"Total Rejected Epochs: {len(rejected_epochs)}")
                st.write(f"Min Epoch: {min(rejected_epochs)}, Max Epoch: {max(rejected_epochs)}")

                # 可视化分布
                st.subheader("Visualization of Rejected Epochs")
                full_range = pd.DataFrame({"Epoch": range(0, num_epochs + 1)})
                rejected_counts = df["Rejected Epochs"].value_counts().reset_index()
                rejected_counts.columns = ["Epoch", "Frequency"]
                full_df = full_range.merge(rejected_counts, on="Epoch", how="left").fillna(0)

                # 转换 Frequency 列为整数
                full_df["Frequency"] = full_df["Frequency"].astype(int)

                # 展示条形图
                st.bar_chart(full_df.set_index("Epoch")["Frequency"])

                # 搜索功能
                st.subheader("Search Specific Epoch")
                search_value = st.number_input(
                    "Search for an epoch:", min_value=min(rejected_epochs), max_value=max(rejected_epochs), value=min(rejected_epochs)
                )
                if search_value in rejected_epochs:
                    st.success(f"Epoch {search_value} is rejected.")
                else:
                    st.warning(f"Epoch {search_value} is not in the rejected list.")
            else:
                st.warning("The log content is not a valid list.")
        except Exception as e:
            st.error(f"Error parsing log content: {e}")
    else:
        st.warning(f"Reject log file not found at {reject_log_file}.")


