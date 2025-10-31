# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import streamlit as st

dataset_report_path = os.getenv('DATASET_REPORT_PATH', '/')
st.session_state.dataset_report_path = dataset_report_path
subjects_dir = os.getenv('SUBJECTS_DIR', '/smri')
st.session_state.subjects_dir = subjects_dir

st.set_page_config(page_title="MEG Prep Reports", layout="wide", page_icon="_static/favicon.png",)
# st.write("## Welcome to MEG Prep Reports!")
# st.sidebar.success("Preprocess")

preproc_page = st.Page("reports/preproc.py", title="Preproc", icon=":material/dashboard:")
ica_page = st.Page("reports/ICA.py", title="ICA", icon=":material/dashboard:")
epochs_page = st.Page("reports/epochs.py", title="Epochs", icon=":material/dashboard:")
covar_page = st.Page("reports/covariance.py", title="Covariance", icon=":material/dashboard:")

headmodel_page = st.Page("reports/headmodel.py", title="Head Model - BEM Surfaces", icon=":material/dashboard:")

coreg_page = st.Page("reports/coreg.py", title="Coregistration", icon=":material/dashboard:")
coreg_page_3d = st.Page("reports/coregistration.py", title="Coregistration [3D]", icon=":material/dashboard:")

source_page = st.Page("reports/source_imaging.py", title="Source Localization", icon=":material/dashboard:")
source_vis_page = st.Page("reports/source_imaging_3d.py", title="Source Localization [3D]", icon=":material/dashboard:")

nextflow_page = st.Page("reports/nextflow.py", title="NextFlow Resources", icon=":material/dashboard:")
nextflow_config_page = st.Page("reports/nx_config_online.py", title="NextFlow Configure", icon=":material/dashboard:")

# anatomy_page = st.Page("reports/anatomy_vis.py", title="Freesurfer Recon-all", icon=":material/add_circle:")

# test_page = st.Page("reports/anatomy.py", title="freesurfer  pysurfer", icon=":material/add_circle:")
quality_check_page = st.Page("reports/quality_check.py",title="Quality Checker", icon=":material/dashboard:")

# search_page = st.Page("tools/search.py", title="Search", icon=":material/search:")
# history = st.Page("tools/history.py", title="History", icon=":material/history:")

pg = st.navigation([preproc_page, ica_page, epochs_page,covar_page,
                    headmodel_page,
                    coreg_page,coreg_page_3d,
                    source_page,source_vis_page,
                    quality_check_page,
                    nextflow_config_page, nextflow_page,
                    ])
# pg = st.navigation([preproc_page, ica_page, coreg_page,nextflow_page])

pg.run()