# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from reports.utils import in_docker,filter_files_by_keyword

# set report root dir.
if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

DEFAULT_ICA_REPORT_DIR = report_root_dir / "preprocessed" / "ica_report"


# Enhanced CSS styling with fixed button sizes
enhanced_style = """
    <style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Global styling */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        }

        /* Main header styling */
        .main-header {
            # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-weight: 700;
            # box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
            margin-bottom: 30px;
            # animation: fadeInDown 0.6s ease-out;
        }

        /* Subheader styling */
        .stSubheader {
            color: #2c3e50;
            font-weight: 600;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            margin-bottom: 20px;
        }

        /* Button styling - FIXED CONSISTENT SIZING */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 20px !important;
            border-radius: 10px;
            font-weight: 600;
            font-size: 14px !important;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100% !important;
            height: 48px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Ensure consistent button container sizing */
        .stButton {
            width: 100%;
        }

        /* Image container styling */
        .stImage {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .stImage:hover {
            transform: scale(1.02);
        }

        /* Marked component cards */
        .marked-card-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-top: 20px;
        }

        .marked-card {
            background: linear-gradient(135deg, #e8f9ee 0%, #d6f5dc 100%);
            border: 2px solid #52c788;
            border-radius: 12px;
            padding: 15px;
            margin: 8px;
            box-shadow: 0 4px 8px rgba(82, 199, 136, 0.2);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .marked-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(82, 199, 136, 0.3);
            border-color: #45b574;
        }

        /* Info box styling */
        .info-box {
            background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
            border-left: 5px solid #ffc107;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(255, 193, 7, 0.2);
        }

        /* Score badge styling */
        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin: 5px;
            box-shadow: 0 2px 4px rgba(79, 172, 254, 0.3);
        }

        /* Success message styling */
        .success-msg {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 5px solid #28a745;
            padding: 15px;
            border-radius: 10px;
            color: #155724;
            font-weight: 600;
            box-shadow: 0 4px 8px rgba(40, 167, 69, 0.2);
            animation: slideInLeft 0.5s ease-out;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }

        /* Component counter badge */
        .component-badge {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: 700;
            display: inline-block;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(240, 147, 251, 0.3);
        }

        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        /* Column spacing */
        .block-container {
            padding: 2rem 3rem;
        }

        /* Save button special styling */
        div[data-testid="column"]:has(button:contains("Save")) button {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
            font-size: 16px !important;
            height: 56px !important;
        }

        div[data-testid="column"]:has(button:contains("Save")) button:hover {
            box-shadow: 0 8px 16px rgba(17, 153, 142, 0.4);
        }

        /* Marked component button styling */
        button[key^="view_"], button[key^="delete_"] {
            height: 44px !important;
            font-size: 13px !important;
        }
    </style>
"""
st.markdown(enhanced_style, unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("""
    <div style='text-align: center; padding: 0px;'>
        <h2 >⚙️ Settings</h2>
    </div>
""", unsafe_allow_html=True)



ica_report_dir = st.sidebar.text_input(
    "📁 ICA Report Directory:",
    value=DEFAULT_ICA_REPORT_DIR
)

image_dirs = sorted([f for f in os.listdir(ica_report_dir)])

# Custom filter
st.sidebar.markdown('<div class="filter-box">', unsafe_allow_html=True)

# Initialize filter keyword in session state
if 'filter_keyword' not in st.session_state:
    st.session_state.filter_keyword = ""

filter_keyword = st.sidebar.text_input(
    "🔍 Filter files by keyword:",
    value=st.session_state.filter_keyword,
    placeholder="e.g., sub-01, task-rest, run-1, etc.",
    help="Enter any keyword to filter files (case-insensitive). Leave empty to show all files.",
    key="filter_input"
)
st.session_state.filter_keyword = filter_keyword

# Filter files
filtered_dirs = filter_files_by_keyword(image_dirs, filter_keyword)

selected_dir = st.sidebar.selectbox("📊 Select a MEG File:", filtered_dirs)
print("selected directory: ", selected_dir)

image_dir = os.path.join(ica_report_dir, selected_dir, 'ica_results')
MARKED_FILE = os.path.join(ica_report_dir, selected_dir, "marked_components.txt")
print("MARKED_FILE:", MARKED_FILE)
ecg_eog_score_file = os.path.join(ica_report_dir, selected_dir, "ecg_eog_scores.json")

# Streamlit title
st.markdown('<h2 class="main-header">🧠 Interactive ICA Component Viewer/Marker</h2>', unsafe_allow_html=True)

# init session
if "last_selected_dir" not in st.session_state:
    st.session_state["last_selected_dir"] = selected_dir
if "ica_component" not in st.session_state:
    st.session_state["ica_component"] = 0
if "source_group" not in st.session_state:
    st.session_state["source_group"] = 0
if 'marked_types' not in st.session_state:
    st.session_state['marked_types'] = []
if "marked_components" not in st.session_state:
    if os.path.exists(MARKED_FILE):
        with open(MARKED_FILE, "r") as f:
            st.session_state["marked_components"] = [int(line.strip()) for line in f.readlines()]
    else:
        st.session_state["marked_components"] = []
if 'ecg_eog_scores' not in st.session_state:
    # ECG & EOG Scores
    if os.path.exists(ecg_eog_score_file):
        with open(ecg_eog_score_file, 'r', encoding='utf-8') as f:
            ecg_eog_scores = json.load(f)
        st.session_state['ecg_eog_scores'] = ecg_eog_scores
    else:
        st.session_state['ecg_eog_scores'] = {
            'ecg_indices': [],
            'eog_indices': [],
            'ecg': [],
            'eog': []
        }
if st.session_state["last_selected_dir"] != selected_dir:
    st.session_state["ica_component"] = 0
    st.session_state["source_group"] = 0
    if os.path.exists(MARKED_FILE):
        with open(MARKED_FILE, "r") as f:
            st.session_state["marked_components"] = [int(line.strip()) for line in f.readlines()]
    else:
        st.session_state["marked_components"] = []
    st.session_state["last_selected_dir"] = selected_dir

print("st.session_state:", st.session_state)

if "refresh" not in st.session_state:
    st.session_state["refresh"] = False

if not os.path.exists(image_dir):
    st.error(f"❌ Image directory '{image_dir}' does not exist. Please check the path and try again.")
else:
    files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    if not files:
        st.warning("⚠️ No image files found in the specified directory.")
    else:
        # Parse the topology diagram file
        def parse_filename(filename):
            match = re.search(r"(\d+)_evar_(-?\d+\.?\d*)(?=\.png)", filename)
            if match:
                component_number = int(match.group(1))
                score = float(match.group(2))
                return {"filename": filename, "component": component_number, "ExplainedVar": score}
            return None


        # parse sources file
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
            st.error("❌ Invalid component index. Please check the file structure.")
        elif source_group_idx < 0 or source_group_idx >= len(source_files):
            st.error("❌ Invalid source group index. Please check the file structure.")
        else:
            current_topo = topo_files[component_idx]
            current_topo_filename = current_topo["filename"]
            current_topo_score = current_topo["ExplainedVar"]
            current_eog_score = None
            current_ecg_score = None
            try:
                if component_idx in st.session_state['ecg_eog_scores']['ecg_indices']:
                    pos = st.session_state['ecg_eog_scores']['ecg_indices'].index(component_idx)
                    current_ecg_score = st.session_state['ecg_eog_scores']['ecg'][pos]

                if component_idx in st.session_state['ecg_eog_scores']['eog_indices']:
                    pos = st.session_state['ecg_eog_scores']['eog_indices'].index(component_idx)
                    if st.session_state['ecg_eog_scores']['eog']:
                        current_eog_score = st.session_state['ecg_eog_scores']['eog'][pos]
                    else:
                        current_eog_score = 0.5
            except Exception as e:
                print("ecg_eog_scores error:",e)

            current_source = source_files[source_group_idx]
            current_source_filename = current_source["filename"]
            current_source_start = current_source["start"]
            current_source_end = current_source["end"]

            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown(f"### 📈 Source Components: {current_source_start}-{current_source_end}")
                st.image(
                    os.path.join(image_dir, current_source_filename),
                    caption=f"Source Components {current_source_start}-{current_source_end}",
                    use_container_width=True,
                )

                left_col1, left_col2 = st.columns(2, gap="medium")
                with left_col1:
                    if st.button("Previous Sources", use_container_width=True):
                        st.session_state["source_group"] = max(0, source_group_idx - 1)
                        st.rerun()
                with left_col2:
                    if st.button("Next Sources", use_container_width=True):
                        st.session_state["source_group"] = min(len(source_files) - 1, source_group_idx + 1)
                        st.rerun()

            total_components = len(topo_files)
            with col2:
                st.markdown(f"### 🔬 Component {current_topo['component']}/{total_components - 1} - Topography")

                # Create score badges
                score_html = f'<span class="score-badge">📊 ExplainVar: {current_topo_score:.3f}</span>'
                if current_eog_score is not None:
                    score_html += f'<span class="score-badge">👁️ EOG: {current_eog_score:.2f}</span>'
                if current_ecg_score is not None:
                    score_html += f'<span class="score-badge">❤️ ECG: {current_ecg_score:.2f}</span>'

                st.markdown(f'<div style="margin-bottom: 15px;">{score_html}</div>', unsafe_allow_html=True)

                st.image(
                    os.path.join(image_dir, current_topo_filename),
                    use_container_width=True,
                )

                right_col1, right_col2  = st.columns(2, gap="small")
                with right_col1:
                    if st.button("Previous", use_container_width=True):
                        st.session_state["ica_component"] = max(0, component_idx - 1)
                        st.rerun()
                with right_col2:
                    if st.button("Next ", use_container_width=True):
                        st.session_state["ica_component"] = min(len(topo_files) - 1, component_idx + 1)
                        st.rerun()

                m_col1,m_col2,m_col3 = st.columns(3, gap="small")
                with m_col1:
                    if st.button("Mark", use_container_width=True):
                        if current_topo["component"] not in st.session_state["marked_components"]:
                            st.session_state["marked_components"].append(current_topo["component"])
                            st.session_state["marked_types"].append("outlier") #outlier
                            st.toast(f"✅ Component {current_topo['component']} marked as artifact.")
                with m_col2:
                    if st.button("Mark as ECG", use_container_width=True):
                        if current_topo["component"] not in st.session_state["marked_components"]:
                            st.session_state["marked_components"].append(current_topo["component"])
                            st.session_state["marked_types"].append("ecg")  # Record ECG type
                            if current_topo["component"] not in st.session_state['ecg_eog_scores']['ecg_indices']:
                                st.session_state['ecg_eog_scores']['ecg_indices'].append(current_topo["component"])
                                # For ECG,
                                st.session_state['ecg_eog_scores']['ecg'].append(1.0)
                            st.toast(f"✅ Component {current_topo['component']} marked as ECG.")
                with m_col3:
                    if st.button("Mark as EOG", use_container_width=True):
                        if current_topo["component"] not in st.session_state["marked_components"]:
                            st.session_state["marked_components"].append(current_topo["component"])
                            st.session_state["marked_types"].append("eog")  # Record EOG type
                            if current_topo["component"] not in st.session_state['ecg_eog_scores']['eog_indices']:
                                st.session_state['ecg_eog_scores']['eog_indices'].append(current_topo["component"])
                                # For EOG
                                st.session_state['ecg_eog_scores']['eog'].append(1.0)
                            st.toast(f"✅ Component {current_topo['component']} marked as EOG.")
            print("ecg_eog_scores:sssss",st.session_state['ecg_eog_scores'])
            # Marked components section
            st.markdown(
                "<hr style='margin: 40px 0; border: none; height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent);'>",
                unsafe_allow_html=True)

            st.markdown("### 🏷️ Marked ICA Components")

            if st.session_state["marked_components"]:
                st.markdown(
                    f'<div class="component-badge">📍 Total Marked: {len(st.session_state["marked_components"])}</div>',
                    unsafe_allow_html=True)

                # Each row can show up to 6 items
                items_per_row = 6

                # Group marked components into rows
                rows = [
                    st.session_state["marked_components"][i: i + items_per_row]
                    for i in range(0, len(st.session_state["marked_components"]), items_per_row)
                ]

                # Iterate through each row
                for row in rows:
                    cols = st.columns(len(row), gap="small")
                    for i, comp in enumerate(row):
                        with cols[i]:
                            if st.button(f"📌 Comp {comp}", key=f"view_{comp}", use_container_width=True):
                                st.session_state["ica_component"] = next(
                                    (idx for idx, topo in enumerate(topo_files) if topo["component"] == comp),
                                    st.session_state["ica_component"]
                                )
                                st.rerun()

                            if st.button(f"🗑️ Delete", key=f"delete_{comp}", use_container_width=True):
                                # st.session_state["marked_components"].remove(comp)

                                # Remove from marked types if it exists
                                index_to_remove = None

                                # Find and remove the corresponding type
                                for idx, (m_comp, m_type) in enumerate(
                                        zip(st.session_state["marked_components"], st.session_state["marked_types"])):
                                    if m_comp == comp:
                                        st.session_state["marked_types"].pop(idx)
                                        break  # Found and removed the corresponding type
                                # Update existing scores
                                if comp in st.session_state['ecg_eog_scores']['ecg_indices']:
                                    idx = st.session_state['ecg_eog_scores']['ecg_indices'].index(comp)
                                    st.session_state['ecg_eog_scores']['ecg_indices'].pop(idx)
                                    st.session_state['ecg_eog_scores']['ecg'].pop(idx)  # Remove corresponding ECG score
                                elif comp in st.session_state['ecg_eog_scores']['eog_indices']:
                                    idx = st.session_state['ecg_eog_scores']['eog_indices'].index(comp)
                                    st.session_state['ecg_eog_scores']['eog_indices'].pop(idx)
                                    st.session_state['ecg_eog_scores']['eog'].pop(idx)  # Remove corresponding EOG score

                                st.session_state["marked_components"].remove(comp)
                                st.success(f"✅ Component {comp} removed.")
                                st.rerun()

            else:
                st.info("ℹ️ No components marked yet. Use the 'Mark' button to mark artifacts.")

            # Save button
            st.markdown(
                "<hr style='margin: 40px 0; border: none; height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent);'>",
                unsafe_allow_html=True)

            st.markdown("### 💾 Save ICA Components")
            col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
            with col_save2:
                if st.button("💾 Save Marked Components", use_container_width=True):
                    with open(MARKED_FILE, "w") as f:
                        f.write("\n".join(map(str, sorted(st.session_state["marked_components"]))))

                    with open(ecg_eog_score_file, "w") as json_file:
                        json.dump(st.session_state['ecg_eog_scores'], json_file, indent=4)

                    st.markdown(
                        f'<div class="success-msg">✅ Successfully saved {len(st.session_state["marked_components"])} marked components to:<br><code>{MARKED_FILE}</code></div>',
                        unsafe_allow_html=True)
