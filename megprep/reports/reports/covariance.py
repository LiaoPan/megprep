# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import streamlit as st
from pathlib import Path
from reports.utils import in_docker

# Title of the application
st.title("Covariance Visualization")

if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

covar_report_dir = report_root_dir / "preprocessed" / "covariance"

covar_report_dir = st.sidebar.text_input(
    "Covariance Report Directory:",
    value=str(covar_report_dir)
)

if os.path.exists(covar_report_dir):
    image_dirs = sorted([f for f in os.listdir(covar_report_dir) if os.path.isdir(os.path.join(covar_report_dir, f))])
else:
    image_dirs = []

selected_dir = st.sidebar.selectbox("Select a MEG File:", image_dirs)

st.subheader("Baseline(Noise) Covariance")
st.image(os.path.join(covar_report_dir,selected_dir,f"bl_cov.png"))

st.subheader("Baseline(Noise) Covariance Spectra")
st.image(os.path.join(covar_report_dir,selected_dir,f"bl_cov_spectra.png"))