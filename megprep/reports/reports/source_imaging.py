# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import pyvista as pv
import mne
import tempfile
import altair as alt
from pathlib import Path
from reports.utils import in_docker

# --- 1. 设置环境变量 (在无头环境中避免 GLSL 错误) ---
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
# 如果在无头环境，需要指定 DISPLAY
# virtual GUI |
# Xvfb :99 -screen 0 1920x1080x24 &
os.environ["DISPLAY"] = ":99"

# --- 启动虚拟显示 (Xvfb) ---
if "IS_XVFB_RUNNING" not in st.session_state:
    start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

st.title("MNE Source Localization")

if in_docker():
    report_root_dir = Path("/output")
    subjects_dir = Path("/smri")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))
    subjects_dir = Path(st.session_state.get("subjects_dir"))

source_recon_report_dir = report_root_dir / "preprocessed" / "source_recon"

source_report_dir = st.sidebar.text_input(
    "Source Recon Report Directory:",
    value=str(source_recon_report_dir)
)

if os.path.exists(source_report_dir):
    image_dirs = sorted([f for f in os.listdir(source_report_dir) if os.path.isdir(os.path.join(source_report_dir, f))])
else:
    image_dirs = []

selected_dir = st.sidebar.selectbox("Select a MEG File:", image_dirs)

selected_ori = st.sidebar.radio("Select Brain Hemisphere:", ["Left Hemisphere (lh)", "Right Hemisphere (rh)"])

ori_suffix = "lh" if "Left" in selected_ori else "rh"

if selected_dir:
    source_imaging = Path(source_report_dir) / selected_dir / f"wdonset_evoked_dSPM-ico4-{ori_suffix}.png"

    if source_imaging.exists():
        st.image(source_imaging, use_container_width=True)
    else:
        st.warning(f"Image not found: {source_imaging}")
else:
    st.info("Please select a MEG File.")


## --- STC Source Estimate Visualization ---
st.subheader("Interactive Source Estimate (STC) Display")
st.sidebar.markdown("---")

stc_dir = Path(source_report_dir) / selected_dir if selected_dir else None
if stc_dir and stc_dir.exists():
    # Find all stc file prefixes (excluding hemi)
    stc_files = [f for f in os.listdir(stc_dir) if f.endswith('.stc')]
    prefixes = set()
    for f in stc_files:
        if f.endswith('-lh.stc') or f.endswith('-rh.stc'):
            prefixes.add(f[:-7])
    prefixes = sorted(list(prefixes))
else:
    prefixes = []

if prefixes:
    selected_prefix = st.sidebar.selectbox("Select STC file prefix (auto matches lh/rh)", prefixes)
    stc_path = stc_dir / f"{selected_prefix}-{ori_suffix}.stc"

    # Load and display the STC, with interactive visualization controls
    if stc_path.exists():
        st.info(f"Source Estimate – {selected_prefix}")
        stc = mne.read_source_estimate(str(stc_path))

        selected_ori_hemi = st.sidebar.radio("Brain Hemisphere:", ["lh", "rh","split"])


        # --- Time slider ---
        time_min, time_max = stc.times[0], stc.times[-1]
        default_time = (time_min + time_max) / 2
        time = st.sidebar.slider(
            "Display Time Point (s)", float(time_min), float(time_max),
            value=float(default_time), step=0.01, format="%.3f"
        )

        # --- Views multi-select ---
        views_options = ["lateral", "medial", "dorsal", "ventral", "frontal", "parietal", "axial"]
        views_selected = st.sidebar.multiselect("Brain View(s)", views_options, default=["lateral"])
        views = views_selected if len(views_selected) > 1 else views_selected[0]

        # --- CLIM and display options ---
        background_color = st.sidebar.selectbox("Background",["white","black"])
        clim_kind = st.sidebar.selectbox("clim kind", ["percent", "value"], index=0)
        pos_lims_input = st.sidebar.text_input("clim pos_lims (comma-separated)", value="0, 97.5, 100")
        try:
            pos_lims = [float(x.strip()) for x in pos_lims_input.split(",")][:3]
            if len(pos_lims) != 3:
                raise ValueError
        except Exception:
            st.error("clim pos_lims must be three comma-separated numbers!")
            st.stop()


        time_viewer = st.sidebar.checkbox("time_viewer (interactive timebar)", True)
        show_traces = st.sidebar.checkbox("show_traces (plot time series below)", True)
        if selected_ori_hemi == 'split':
            size = (1200, 500)
        else:
            size = (1000, 800)
        time_label = st.sidebar.text_input("Time label (time_label)", "Time (s)")

        # --- subject options ---
        # List all available subjects (folders) in SUBJECTS_DIR
        if subjects_dir and os.path.exists(subjects_dir):
            subject_choices = sorted(
                [d for d in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, d))]
            )
        else:
            st.warning("SUBJECTS_DIR not found or does not exist!")
            subject_choices = []

        # Streamlit sidebar selectbox for subject
        subject = st.sidebar.selectbox("Select subject for surface visualization", subject_choices)

        surfer_kwargs = dict(
            subject=subject,
            hemi=selected_ori_hemi,
            subjects_dir=subjects_dir,  # supply fsaverage or subjects dir as needed
            clim=dict(kind=clim_kind, pos_lims=pos_lims),
            views=views,
            initial_time=time,
            time_unit='s',
            time_label=time_label,
            alpha=1,
            time_viewer=time_viewer,
            show_traces=show_traces,
            size=size,
            background=background_color,
            smoothing_steps=10,
            brain_kwargs=dict(block=False, show=False),
            verbose=True
        )

        with st.expander("Visualization parameters", expanded=False):
            st.write(f"```{surfer_kwargs}```")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpimg = os.path.join(tmpdir, f"brain_{selected_prefix}_{ori_suffix}_{time:.3f}.png")
            brain = stc.plot(**surfer_kwargs)
            brain.save_image(tmpimg)
            brain.close()
            st.image(tmpimg, caption=f"{selected_prefix}, {ori_suffix}, {time:.3f} s")
    else:
        st.warning(f"STC file not found: {stc_path}")
else:
    st.info("No STC files found for interactive display in this MEG directory.")