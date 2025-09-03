# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from reports.utils import in_docker

# set report root dir.
if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

DEFAULT_ICA_REPORT_DIR = report_root_dir / "preprocessed" / "ica_report"

ica_report_dir = st.sidebar.text_input(
    "ICA Report Directory:",
    value=DEFAULT_ICA_REPORT_DIR
)

image_dirs = sorted([f for f in os.listdir(ica_report_dir)])
selected_dir = st.sidebar.selectbox("Select a MEG File:", image_dirs)
print("selected directory: ", selected_dir)

image_dir = os.path.join(ica_report_dir, selected_dir, 'ica_results')  
MARKED_FILE = os.path.join(ica_report_dir, selected_dir, "marked_components.txt") 
print("MARKED_FILE:",MARKED_FILE)

# ECG & EOG Socres
ecg_eog_score_file = os.path.join(ica_report_dir, selected_dir, "ecg_eog_scores.json")
with open(ecg_eog_score_file, 'r', encoding='utf-8') as f:
    ecg_eog_scores = json.load(f)

# Add CSS to hide the header and footer, and enhance the styling.
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .delete-btn {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .delete-btn:hover {
            background-color: #e63e3e;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Streamlit title
st.title("Interactive ICA Component Viewer and Marker")

# init session
if "last_selected_dir" not in st.session_state:
    st.session_state["last_selected_dir"] = selected_dir
if "ica_component" not in st.session_state:
    st.session_state["ica_component"] = 0  
if "source_group" not in st.session_state:
    st.session_state["source_group"] = 0  
if "marked_components" not in st.session_state:
    
    if os.path.exists(MARKED_FILE):
        with open(MARKED_FILE, "r") as f:
            st.session_state["marked_components"] = [int(line.strip()) for line in f.readlines()]
    else:
        st.session_state["marked_components"] = []  

if st.session_state["last_selected_dir"] != selected_dir:
    st.session_state["ica_component"] = 0
    st.session_state["source_group"] = 0
    if os.path.exists(MARKED_FILE):
        with open(MARKED_FILE, "r") as f:
            st.session_state["marked_components"] = [int(line.strip()) for line in f.readlines()]
    else:
        st.session_state["marked_components"] = []  
   
    st.session_state["last_selected_dir"] = selected_dir
    # st.rerun()
print("st.session_state:",st.session_state)

if "refresh" not in st.session_state:
    st.session_state["refresh"] = False  


if not os.path.exists(image_dir):
    st.error(f"Image directory '{image_dir}' does not exist. Please check the path and try again.")
else:
    files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    if not files:
        st.warning("No image files found in the specified directory.")
    else:
        # Parse the topology diagram file (extract component IDs and additional information)
        def parse_filename(filename):
            # match = re.search(r"ica_(\d+)_k_([\d+\.\d+]+(?=\.png))", filename)
            match = re.search(r"(\d+)_evar_([\d+\.\d+]+(?=\.png))", filename)
            if match:
                component_number = int(match.group(1))
                score = float(match.group(2))
                return {"filename": filename, "component": component_number, "ExplainedVar": score}
            return None

        # parse sources file（ ica_comp_20-39_tc.png）
        def parse_source_group(filename):
            match = re.search(r"ica_comp_(\d+)-(\d+)_tc", filename)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                return {"filename": filename, "start": start, "end": end}
            return None

        topo_files = [parse_filename(f) for f in files if parse_filename(f)]
        topo_files = sorted(topo_files, key=lambda x: x["component"]) 

        source_files = [parse_source_group(f) for f in files if parse_source_group(f)]
        source_files = sorted(source_files, key=lambda x: x["start"]) 

        component_idx = st.session_state["ica_component"]
        source_group_idx = st.session_state["source_group"]

        if component_idx < 0 or component_idx >= len(topo_files):
            st.error("Invalid component index. Please check the file structure.")
        elif source_group_idx < 0 or source_group_idx >= len(source_files):
            st.error("Invalid source group index. Please check the file structure.")
        else:
            current_topo = topo_files[component_idx]
            current_topo_filename = current_topo["filename"]
            current_topo_score = current_topo["ExplainedVar"]
            current_eog_score = None
            current_ecg_score = None
            try:
                if component_idx in ecg_eog_scores['ecg_indices']:
                    pos = ecg_eog_scores['ecg_indices'].index(component_idx)
                    current_ecg_score = ecg_eog_scores['ecg'][pos]

                if component_idx in ecg_eog_scores['eog_indices']:
                    pos = ecg_eog_scores['eog_indices'].index(component_idx)
                    current_eog_score = ecg_eog_scores['eog'][pos]
            except Exception as e:
                st.error(e)

            current_source = source_files[source_group_idx]
            current_source_filename = current_source["filename"]
            current_source_start = current_source["start"]
            current_source_end = current_source["end"]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(f"Source Components: {current_source_start}-{current_source_end}")
                st.image(
                    os.path.join(image_dir, current_source_filename),
                    caption=f"Source Components {current_source_start}-{current_source_end}",
                    use_container_width=True,
                )

                left_col1, left_col2 = st.columns(2)
                with left_col1:
                    if st.button("Previous Sources Group"):
                        st.session_state["source_group"] = max(0, source_group_idx - 1)
                        st.rerun()
                with left_col2:
                    if st.button("Next Sources Group"):
                        st.session_state["source_group"] = min(len(source_files) - 1, source_group_idx + 1)
                        st.rerun()

            total_components = len(topo_files)
            with col2:
                st.subheader(f"Component {current_topo['component']}/{total_components-1} - Topography")
                content = f"ExplainVar Score: {current_topo_score}"
                
                if current_eog_score is not None:
                    content += f"| EOG score: {current_eog_score}"
                if current_ecg_score is not None:
                    content += f"| ECG score: {current_ecg_score}"

                st.image(
                    os.path.join(image_dir, current_topo_filename),
                    caption= content,
                    use_container_width=True,
                )

                right_col1, right_col2, right_col3 = st.columns(3)
                with right_col1:
                    if st.button("Previous Component"):
                        st.session_state["ica_component"] = max(0, component_idx - 1)
                        st.rerun()
                        # st.session_state["refresh"] = not st.session_state["refresh"]  
                with right_col2:
                    if st.button("Next Component"):
                        # print("component_idx",component_idx)
                        st.session_state["ica_component"] = min(len(topo_files) - 1, component_idx + 1)
                        st.rerun()
                        # st.session_state["refresh"] = not st.session_state["refresh"] 
                with right_col3:
                    # Display "Mark Component" button
                    if st.button("Mark Component"):
                        # print("current_topo[component]", current_topo["component"])
                        # print("st.session_state[marked_components]", st.session_state["marked_components"])
                        if current_topo["component"] not in st.session_state["marked_components"]:
                            st.session_state["marked_components"].append(current_topo["component"])
                            st.success(f"Component {current_topo['component']} marked as artifact.")
                            # st.session_state["refresh"] = not st.session_state["refresh"]  # 刷新页面

            st.subheader("Marked ICA Components")

            if st.session_state["marked_components"]:
                # Each row can show up to 6 items
                items_per_row = 6

                # Custom card and button styles
                card_style = """
                    <style>
                        .marked-card {
                            background-color: #e8f9ee;
                            border: 1px solid #ddd;
                            border-radius: 10px;
                            padding: 10px;
                            margin: 5px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;
                            width: 120px;
                            text-align: center;
                            color: black;
                            cursor: pointer;
                        }
                        .marked-card:hover {
                            background-color: #d6f5dc;
                        }
                        .delete-btn {
                            background-color: #ff4b4b;
                            color: white;
                            border: none;
                            padding: 5px 10px;
                            border-radius: 5px;
                            cursor: pointer;
                            font-size: 12px;
                            margin-top: 10px;
                        }
                        .delete-btn:hover {
                            background-color: #e63e3e;
                        }
                    </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)

                # Group marked components into rows
                rows = [
                    st.session_state["marked_components"][i: i + items_per_row]
                    for i in range(0, len(st.session_state["marked_components"]), items_per_row)
                ]

                # Iterate through each row
                for row in rows:
                    cols = st.columns(len(row))  # Dynamically create columns for each row
                    for i, comp in enumerate(row):
                        with cols[i]:
                            # Display card with delete button
                            if st.button(f"Component {comp}", key=f"view_{comp}"):
                                st.session_state["ica_component"] = next(
                                    (idx for idx, topo in enumerate(topo_files) if topo["component"] == comp),
                                    st.session_state["ica_component"]
                                )
                                st.rerun()

                            # Display delete button
                            if st.button(f"Delete", key=f"delete_{comp}"):
                                st.session_state["marked_components"].remove(comp)
                                st.success(f"Component {comp} removed.")
                                st.rerun()

            else:
                st.write("No components marked yet.")

            # Save button
            st.subheader("Save ICA Components")
            if st.button("Save Marked Components"):
                with open(MARKED_FILE, "w") as f:
                    f.write("\n".join(map(str, sorted(st.session_state["marked_components"]))))
                st.success(f"Marked components saved to {MARKED_FILE}!")


