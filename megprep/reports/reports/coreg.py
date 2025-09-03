# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import streamlit as st
from pathlib import Path
import pandas as pd
from reports.utils import in_docker

st.write("## MEG Preprocessing Results Viewer")

if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

coreg_report_dir = report_root_dir / "preprocessed" / "trans"
image_dirs = sorted([f for f in os.listdir(coreg_report_dir)])
selected_dir = st.sidebar.selectbox("Select a MEG File:", image_dirs) # subject id
coreg_dir = coreg_report_dir / selected_dir

dists = pd.read_csv(coreg_dir / "dists.csv", index_col=None)

st.write("## Coregistration Distance")
st.dataframe(dists)

st.write("## Coregistration Visualization")
for img in coreg_dir.glob("*.png"):
    st.image(img, caption=img.name)

