# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import streamlit as st
from pathlib import Path
from reports.utils import in_docker

# Title of the application
st.title("Head Model Visualization")

if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

fwd_report_dir = report_root_dir / "preprocessed" / "forward_solution"

fwd_report_dir = st.sidebar.text_input(
    "Head Model Report Directory:",
    value=str(fwd_report_dir)
)

if os.path.exists(fwd_report_dir):
    image_dirs = sorted([f for f in os.listdir(fwd_report_dir) if os.path.isdir(os.path.join(fwd_report_dir, f))])
else:
    image_dirs = []

selected_dir = st.sidebar.selectbox("Select a MEG File:", image_dirs)

orientations = ['coronal', 'sagittal', 'axial']
for orientation in orientations:
    st.subheader(orientation)
    st.image(os.path.join(fwd_report_dir,selected_dir,f"headmodel_{orientation}.png"))
