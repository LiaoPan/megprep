# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
from streamlit.components.v1 import html
from reports.utils import in_docker
from pathlib import Path

if in_docker():
    DEFAULT_NX_REPORT_DIR = Path("/output")
else:
    DEFAULT_NX_REPORT_DIR = Path(st.session_state.get("dataset_report_path"))

# Define the path for the nextflow.config file
nx_file_path = Path(st.sidebar.text_input(
    "NextFlow Resource Report Directory:",
    value=DEFAULT_NX_REPORT_DIR
))

nx_report_file_path = nx_file_path / "report.html"
nx_timeline_file_path = nx_file_path / "timeline.html"

flag = ['Resource', 'Timeline']
for idx,ht in enumerate([nx_report_file_path, nx_timeline_file_path]):
    with open(ht, "r") as file:
        html_content = file.read()

    st.title(f"Nextflow {flag[idx]} Reports")
    html(html_content, height=4500, scrolling=True)
