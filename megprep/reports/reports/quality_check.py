#coding:utf-8
import streamlit as st
from reports.utils import in_docker
from pathlib import Path
import os
# Preprocess
# set report root dir.
if in_docker():
    report_root_dir = "/output"
else:
    report_root_dir = st.session_state.get("dataset_report_path")

DATA_DIR = os.path.join(report_root_dir, "preprocessed")


# ICA
# set report root dir.
if in_docker():
    report_root_dir = Path("/output")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))

DEFAULT_ICA_REPORT_DIR = report_root_dir / "preprocessed" / "ica_report"



# Trans
if in_docker():
    report_root_dir = Path("/output")
    default_subjects_dir = Path("/smri")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))
    default_subjects_dir = Path(st.session_state.get("subjects_dir"))

default_meg_dir = report_root_dir / "preprocessed"
default_trans_dir = report_root_dir / "preprocessed" / "trans"

import streamlit as st
import pandas as pd
from pathlib import Path


def matrices_approx_equal(m1, m2, tol=1e-6):
    """Check if two 4x4 matrices are approximately equal within tolerance"""
    if not m1 or not m2:
        return False
    for row1, row2 in zip(m1, m2):
        for a, b in zip(row1, row2):
            if abs(a - b) > tol:
                return False
    return True


def parse_matrix(txt):
    """Parse matrix string"""
    try:
        rows = txt.strip().split('\n')
        return [[float(x) for x in row.split(',')] for row in rows]
    except Exception:
        return None


def load_meg_data(file_path):
    """
    Simulate loading MEG file data
    In production, replace with actual file reading logic (e.g., MNE-Python)
    Returns data in dictionary format
    """
    # Using simulated data here, should read real files in production
    import random
    random.seed(hash(file_path))

    return {
        'ica_ecg': [0.15, 0.18] if random.random() > 0.3 else None,
        'ica_eog': [0.25, 0.10] if random.random() > 0.2 else None,
        'total_channels': 306,
        'bad_channels': random.randint(5, 50),
        'total_segments': 500,
        'bad_segments': random.randint(10, 100),
        'initial_trans': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        'final_trans': [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] if random.random() > 0.4 else [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }


def check_meg_file(data, check_settings):
    """
    Check a single MEG file data based on enabled check settings
    Returns list of alarms
    """
    alarms = []

    # 1. ICA component check
    if check_settings['check_ica_ecg']:
        if not data.get('ica_ecg'):
            alarms.append(("ICA", "ECG component missing"))

    if check_settings['check_ica_eog']:
        if not data.get('ica_eog'):
            alarms.append(("ICA", "EOG component missing"))

    # 2. Artifact detection check - Bad Channels
    if check_settings['check_bad_channels']:
        bad_channels = data.get('bad_channels', 0)
        bad_channel_threshold = check_settings['bad_channel_threshold']

        if bad_channels > bad_channel_threshold:
            alarms.append(("Artifacts Detection",
                           f"Too many bad channels: {bad_channels} (threshold: {bad_channel_threshold})"))

    # 3. Artifact detection check - Bad Segments
    if check_settings['check_bad_segments']:
        bad_segments = data.get('bad_segments', 0)
        bad_segment_threshold = check_settings['bad_segment_threshold']

        if bad_segments > bad_segment_threshold:
            alarms.append(("Artifacts Detection",
                           f"Too many bad segments: {bad_segments} (threshold: {bad_segment_threshold})"))

    # 4. Coregistration check
    if check_settings['check_coregistration']:
        initial_trans = data.get('initial_trans')
        final_trans = data.get('final_trans')

        if initial_trans and final_trans:
            if matrices_approx_equal(initial_trans, final_trans):
                alarms.append(
                    ("Coregistration", "Initial and final coregistration matrices are identical (no fine-tuning)"))
        else:
            alarms.append(("Coregistration", "Coregistration matrix data missing or cannot be parsed"))

    return alarms


st.title("🧠 Report Checker")
st.markdown(
    """
    Supports quality control checks for multiple MEG files
    - 📋 **ICA Component Extraction Completeness**
    - 🔍 **Artifact Detection Bad Channel/Segment Thresholds**
    - 🎯 **Coregistration Matrix Fine-tuning**
    """
)

# Sidebar: File Import Configuration & Settings
with st.sidebar:
    st.header("📁 File Import")

    input_mode = st.radio(
        "Select Import Method",
        ["Text Input File Paths", "Generate Test Files"]
    )

    file_list = []

    if input_mode == "Text Input File Paths":
        file_paths = st.text_area(
            "Enter file paths (one per line)",
            height=150,
            placeholder="/path/to/meg_file1.fif\n/path/to/meg_file2.fif\n..."
        )
        if file_paths:
            file_list = [line.strip() for line in file_paths.split('\n') if line.strip()]

    else:  # Generate test files
        num_files = st.slider("Number of test files to generate", 1, 100, 20)
        if st.button("Generate Test Files"):
            file_list = [f"subject_{i:03d}_meg.fif" for i in range(1, num_files + 1)]
            st.session_state['file_list'] = file_list

    if 'file_list' in st.session_state:
        file_list = st.session_state['file_list']
    else:
        st.session_state['file_list'] = file_list

    # Separator
    st.markdown("---")

    # Check Settings
    st.header("🔍 Check Settings")

    # ICA Checks
    with st.expander("📋 ICA Components", expanded=True):
        st.caption("Flag if component is missing:")
        check_ica_ecg = st.checkbox("Check ECG Component", value=True, key="check_ica_ecg",
                                    help="Flag if ECG component is missing")
        check_ica_eog = st.checkbox("Check EOG Component", value=True, key="check_ica_eog",
                                    help="Flag if EOG component is missing")

    # Bad Channels Check
    with st.expander("🔴 Bad Channels", expanded=True):
        check_bad_channels = st.checkbox("Enable Bad Channels Check", value=True, key="check_bad_channels")
        if check_bad_channels:
            bad_channel_threshold = st.number_input(
                "Maximum allowed bad channels",
                min_value=0,
                max_value=500,
                value=30,
                step=1,
                help="Flag if number of bad channels exceeds this value",
                key="bad_channel_threshold"
            )
            st.caption(f"⚠️ Will flag if bad channels > {bad_channel_threshold}")
        else:
            bad_channel_threshold = 30

    # Bad Segments Check
    with st.expander("📊 Bad Segments", expanded=True):
        check_bad_segments = st.checkbox("Enable Bad Segments Check", value=True, key="check_bad_segments")
        if check_bad_segments:
            bad_segment_threshold = st.number_input(
                "Maximum allowed bad segments",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                help="Flag if number of bad segments exceeds this value",
                key="bad_segment_threshold"
            )
            st.caption(f"⚠️ Will flag if bad segments > {bad_segment_threshold}")
        else:
            bad_segment_threshold = 50

    # Coregistration Check
    with st.expander("🎯 Coregistration", expanded=True):
        check_coregistration = st.checkbox("Enable Coregistration Check", value=True, key="check_coregistration")
        if check_coregistration:
            st.caption("Checks if initial and final transformation matrices are identical")

    # Collect all check settings
    check_settings = {
        'check_ica_ecg': check_ica_ecg,
        'check_ica_eog': check_ica_eog,
        'check_bad_channels': check_bad_channels,
        'bad_channel_threshold': bad_channel_threshold,
        'check_bad_segments': check_bad_segments,
        'bad_segment_threshold': bad_segment_threshold,
        'check_coregistration': check_coregistration,
    }

    # Separator
    st.markdown("---")

    # Display Settings
    st.header("⚙️ Display Settings")

    show_mode = st.selectbox(
        "Display Mode",
        ["All Files", "Files with Alarms Only", "Files without Alarms Only"]
    )

    page_size = st.selectbox("Items per Page", [10, 20, 50, 100], index=1)

    # Separator
    # st.markdown("---")

    # Actions
    # st.header("🔧 Actions")
    #
    # if st.button("🔄 Recheck All Files", use_container_width=True):
    #     st.session_state['check_results'] = {}
    #     st.rerun()

    # Summary of active checks
    st.markdown("---")
    st.caption("**Active Checks:**")
    active_checks = []

    if check_ica_ecg:
        active_checks.append("ICA ECG")
    if check_ica_eog:
        active_checks.append("ICA EOG")
    if check_bad_channels:
        active_checks.append(f"Bad Channels (max: {bad_channel_threshold})")
    if check_bad_segments:
        active_checks.append(f"Bad Segments (max: {bad_segment_threshold})")
    if check_coregistration:
        active_checks.append("Coregistration")

    if active_checks:
        for check in active_checks:
            st.caption(f"✓ {check}")
    else:
        st.caption("⚠️ No checks enabled")

if not file_list:
    st.info("👈 Please import file list in the sidebar")
    st.stop()

# Initialize check results cache
if 'check_results' not in st.session_state:
    st.session_state['check_results'] = {}

# Check if settings have changed (to trigger recheck)
settings_key = str(check_settings)
if 'last_settings' not in st.session_state or st.session_state['last_settings'] != settings_key:
    st.session_state['check_results'] = {}
    st.session_state['last_settings'] = settings_key

# Display total file count
st.markdown(f"### 📊 File Statistics")

# Batch check (lazy loading)
results_summary = []

for file_path in file_list:
    # Check if already cached
    if file_path not in st.session_state['check_results']:
        # Lazy loading: actually read and check file here
        data = load_meg_data(file_path)
        alarms = check_meg_file(data, check_settings)
        st.session_state['check_results'][file_path] = {
            'data': data,
            'alarms': alarms
        }

    result = st.session_state['check_results'][file_path]
    alarm_count = len(result['alarms'])

    results_summary.append({
        'file_path': file_path,
        'alarm_count': alarm_count,
        'has_alarms': alarm_count > 0,
        'alarms': result['alarms']
    })

# Calculate statistics
total_files = len(file_list)
total_with_alarms = sum(1 for r in results_summary if r['has_alarms'])
total_passed = total_files - total_with_alarms
pass_rate = (total_passed / total_files * 100) if total_files > 0 else 0

# Display statistics with pass rate
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Files", total_files)
col2.metric("Passed Files", total_passed, delta_color="normal")
col3.metric("Files with Alarms", total_with_alarms, delta_color="inverse")
col4.metric("Pass Rate", f"{pass_rate:.1f}%", delta_color="normal")

# Apply filters
filtered_results = results_summary.copy()
if show_mode == "Files with Alarms Only":
    filtered_results = [r for r in filtered_results if r['has_alarms']]
elif show_mode == "Files without Alarms Only":
    filtered_results = [r for r in filtered_results if not r['has_alarms']]

# Display filtered count if different from total
if len(filtered_results) != total_files:
    st.info(f"📋 Showing {len(filtered_results)} of {total_files} files based on filter settings")

# Pagination logic
total_pages = (len(filtered_results) + page_size - 1) // page_size if len(filtered_results) > 0 else 1

if total_pages > 1:
    page = st.selectbox(f"Page (Total: {total_pages})", range(1, total_pages + 1))
else:
    page = 1

start_idx = (page - 1) * page_size
end_idx = min(start_idx + page_size, len(filtered_results))
page_results = filtered_results[start_idx:end_idx]

# Display check results
st.markdown("---")
st.subheader(f"Check Results (Page {page}/{total_pages})")

if len(page_results) == 0:
    st.warning("No files match the current filter settings.")
else:
    for idx, result in enumerate(page_results, start=start_idx + 1):
        file_name = Path(result['file_path']).name
        alarm_count = result['alarm_count']

        # Status icon
        if alarm_count == 0:
            status_icon = "✅"
            status_color = "green"
        elif alarm_count <= 2:
            status_icon = "⚠️"
            status_color = "orange"
        else:
            status_icon = "❌"
            status_color = "red"

        # Expandable file details
        with st.expander(f"{status_icon} **{idx}. {file_name}** - {alarm_count} alarm(s)",
                         expanded=(alarm_count > 0 and idx <= start_idx + 3)):

            if alarm_count == 0:
                st.success("✨ All checks passed, no alarms!")
            else:
                st.error(f"Detected {alarm_count} alarm(s):")

                # Display grouped by category
                alarm_by_category = {}
                for category, description in result['alarms']:
                    if category not in alarm_by_category:
                        alarm_by_category[category] = []
                    alarm_by_category[category].append(description)

                for category, descriptions in alarm_by_category.items():
                    st.markdown(f"**{category}**")
                    for desc in descriptions:
                        st.markdown(f"- {desc}")

            # Use toggle instead of nested expander (key fix)
            st.markdown("---")
            show_raw_data = st.toggle("📊 View Raw Data", key=f"toggle_raw_{idx}")
            if show_raw_data:
                data = st.session_state['check_results'][result['file_path']]['data']
                st.json(data)

# Export report
st.markdown("---")
st.subheader("📥 Export Report")

col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"Export {len(filtered_results)} file(s) based on current filter settings")
with col2:
    if st.button("Download CSV", type="primary", use_container_width=True):
        df_export = pd.DataFrame([
            {
                'File Path': r['file_path'],
                'Alarm Count': r['alarm_count'],
                'Status': 'Passed' if r['alarm_count'] == 0 else 'Alarms',
                'Alarm Details': '; '.join([f"[{cat}] {desc}" for cat, desc in r['alarms']])
            }
            for r in filtered_results
        ])
        csv = df_export.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Export Complete Report (CSV)",
            data=csv,
            file_name="meg_report_check.csv",
            mime="text/csv",
            use_container_width=True
        )



