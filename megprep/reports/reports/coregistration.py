# !/usr/bin/env python3
# -*- coding: utf-8 -*-
## ref: https://mne.tools/stable/auto_tutorials/forward/20_source_alignment.html#sphx-glr-auto-tutorials-forward-20-source-alignment-py

import os
import glob
import time
import numpy as np
import nibabel as nib
import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import pyvista as pv
import mne
from pathlib import Path
from scipy import linalg
from mne.io.constants import FIFF
from mne.coreg import Coregistration
from reports.utils import in_docker

# --- 1. 设置环境变量 (在无头环境中避免 GLSL 错误) ---
os.environ['PYVISTA_PLOT_THEME'] = 'document'
os.environ['PYVISTA_USE_PANEL'] = 'False'
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
os.environ['VTK_OFFSCREEN_RENDERING'] = '1'
os.environ["DISPLAY"] = ":99"

# --- 2. 启动虚拟显示 (Xvfb) ---
if "IS_XVFB_RUNNING" not in st.session_state:
    start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True


# --- Custom CSS Styling ---
def apply_custom_styles():
    st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }

        /* Title styling */
        h1 {
            color: #1f77b4;
            font-weight: 700;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #1f77b4;
            margin-bottom: 2rem;
        }

        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }

        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
            color: #2c3e50;
        }

        /* Section headers */
        .section-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Info box styling */
        .info-box {
            background-color: #e8f4f8;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }

        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }

        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }

        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }

        .metric-title {
            color: #6c757d;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            color: #2c3e50;
            font-size: 1.0rem;
            font-weight: 1000;
        }

        /* Transform matrix styling */
        .stTable {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        /* Selectbox styling */
        .stSelectbox > div > div {
            background-color: white;
            border-radius: 6px;
            border: 2px solid #e9ecef;
        }

        /* Divider */
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #e9ecef;
        }

        /* Legend items */
        .legend-item {
            display: inline-block;
            margin-right: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 0.5rem;
            vertical-align: middle;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .legend-text {
            vertical-align: middle;
            font-weight: 500;
            color: #2c3e50;
        }
    </style>
    """, unsafe_allow_html=True)


def load_surface(subject, subjects_dir, trans, surf='white'):
    """
    Load surface data for the left and right hemispheres of the brain.

    Parameters:
    - subject: str, name of the subject
    - surf: str, surface type (e.g., 'white', 'pial', 'inflated', etc.)

    Returns:
    - meshes: dict, contains PyVista PolyData objects for the left and right hemispheres
    """
    hemispheres = ['lh', 'rh']
    meshes = {}

    for hemi in hemispheres:
        surface_path = Path(subjects_dir) / subject / "surf" / f"{hemi}.{surf}"
        coords, faces = mne.read_surface(surface_path)
        coords = mne.transforms.apply_trans(trans, coords, move=True)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(coords, faces)
        meshes[hemi] = mesh
    return meshes


def load_t1(subject, subjects_dir):
    """
    Load the T1 MRI file and change the header information to the correct units.

    Parameters:
    data_path (str or Path): The path to the directory containing the MRI file.

    Returns:
    nibabel.MGHImage: The updated T1 MRI image.
    """
    t1w = nib.load(Path(subjects_dir) / subject / "mri" / "T1.mgz")
    t1w = nib.Nifti1Image(t1w.get_fdata(), t1w.affine)
    t1w.header["xyzt_units"] = np.array(10, dtype="uint8")
    t1_mgh = nib.MGHImage(t1w.get_fdata().astype(np.float32), t1w.affine)
    return t1_mgh


def visualize_head_surface(subject, subjects_dir, trans, raw, t1_mgh, window_size=(800, 600),
                           background_color='white', opacity=0.95):
    """
    Visualize the head surface for a given subject in different coordinate frames.
    """
    surface_path = Path(subjects_dir) / subject / "surf" / "lh.seghead"
    coords, faces = mne.read_surface(surface_path)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(coords, faces)

    mri_to_head = linalg.inv(trans["trans"])
    scalp_pts_in_head_coord = mne.transforms.apply_trans(mri_to_head, coords, move=True)

    head_to_meg = linalg.inv(raw.info["dev_head_t"]["trans"])
    scalp_pts_in_meg_coord = mne.transforms.apply_trans(head_to_meg, scalp_pts_in_head_coord, move=True)

    vox_to_mri = t1_mgh.header.get_vox2ras_tkr()
    mri_to_vox = linalg.inv(vox_to_mri)
    scalp_points_in_vox = mne.transforms.apply_trans(mri_to_vox, coords, move=True)

    meshes = load_surface(subject, subjects_dir, trans=mri_to_vox)

    plotter = pv.Plotter(window_size=window_size, notebook=False)
    plotter.add_mesh(pv.PolyData(scalp_points_in_vox, faces), color="gray", opacity=opacity)
    plotter.add_mesh(meshes['lh'], color="gray", cmap="bwr", show_edges=False, opacity=1, name="lh_brain")
    plotter.add_mesh(meshes['rh'], color="gray", cmap="bwr", show_edges=False, opacity=1, name="rh_brain")

    plotter.set_background(background_color)
    plotter.view_isometric()

    return plotter


def visualize_nasion_and_scalp(plotter, subject, subjects_dir, trans, raw, t1_mgh):
    seghead_rr, seghead_tri = mne.read_surface(Path(subjects_dir) / subject / "surf" / "lh.seghead")

    vox_to_mri = t1_mgh.header.get_vox2ras_tkr()
    mri_to_vox = linalg.inv(vox_to_mri)

    nasion = [
        p for p in raw.info["dig"]
        if p["kind"] == FIFF.FIFFV_POINT_CARDINAL and p["ident"] == FIFF.FIFFV_POINT_NASION
    ][0]
    assert nasion["coord_frame"] == FIFF.FIFFV_COORD_HEAD
    nasion = nasion["r"]

    print("nasion:", nasion)
    nasion_mri = mne.transforms.apply_trans(trans, nasion, move=True)
    nasion_vox = mne.transforms.apply_trans(mri_to_vox, nasion_mri * 1e3, move=True)
    print("nasion vox:", nasion_vox)

    lpa = [
        p for p in raw.info["dig"]
        if p["kind"] == FIFF.FIFFV_POINT_CARDINAL and p["ident"] == FIFF.FIFFV_POINT_LPA
    ][0]
    assert lpa["coord_frame"] == FIFF.FIFFV_COORD_HEAD
    lpa = lpa["r"]
    lpa_mri = mne.transforms.apply_trans(trans, lpa, move=True)
    lpa_vox = mne.transforms.apply_trans(mri_to_vox, lpa_mri * 1e3, move=True)

    rpa = [
        p for p in raw.info["dig"]
        if p["kind"] == FIFF.FIFFV_POINT_CARDINAL and p["ident"] == FIFF.FIFFV_POINT_RPA
    ][0]
    assert rpa["coord_frame"] == FIFF.FIFFV_COORD_HEAD
    rpa = rpa["r"]
    rpa_mri = mne.transforms.apply_trans(trans, rpa, move=True)
    rpa_vox = mne.transforms.apply_trans(mri_to_vox, rpa_mri * 1e3, move=True)

    hsp_points = [p for p in raw.info["dig"] if p["kind"] == FIFF.FIFFV_POINT_EXTRA]
    hsp_vox = []
    for hsp in hsp_points:
        assert hsp["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        hsp_xyz = hsp["r"]
        hsp_mri = mne.transforms.apply_trans(trans, hsp_xyz, move=True)
        hsp_vox.append(mne.transforms.apply_trans(mri_to_vox, hsp_mri * 1e3, move=True))

    hpi_points = [p for p in raw.info["dig"] if p["kind"] == FIFF.FIFFV_POINT_HPI]
    hpi_vox = []
    for hpi in hpi_points:
        assert hpi["coord_frame"] == FIFF.FIFFV_COORD_HEAD
        hpi_xyz = hpi["r"]
        hpi_mri = mne.transforms.apply_trans(trans, hpi_xyz, move=True)
        hpi_vox.append(mne.transforms.apply_trans(mri_to_vox, hpi_mri * 1e3, move=True))

    plotter.add_mesh(pv.Sphere(center=nasion_vox, radius=3, theta_resolution=20, phi_resolution=20),
                     color="orange", opacity=1)
    plotter.add_mesh(pv.Sphere(center=lpa_vox, radius=3, theta_resolution=20, phi_resolution=20),
                     color="blue", opacity=1)
    plotter.add_mesh(pv.Sphere(center=rpa_vox, radius=3, theta_resolution=20, phi_resolution=20),
                     color="red", opacity=1)

    for idx, hpi in enumerate(hpi_vox):
        color = "purple"
        plotter.add_mesh(pv.Sphere(center=hpi, radius=3, theta_resolution=20, phi_resolution=20),
                         color=color, opacity=1)

    for idx, hsp in enumerate(hsp_vox):
        color = "salmon"
        plotter.add_mesh(pv.Sphere(center=hsp, radius=2, theta_resolution=15, phi_resolution=20),
                         color=color, opacity=1)

    return plotter


def rotation_matrix(axis, angle):
    """
    Generate a 3D rotation matrix for a specified axis and angle.

    Parameters:
    -----------
    axis : str
        Rotation axis, must be 'x', 'y', or 'z'
    angle : float
        Rotation angle in degrees

    Returns:
    --------
    numpy.ndarray
        3x3 rotation matrix representing rotation around the specified axis
    """
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


## Main Function

# Apply custom styles
apply_custom_styles()

# Main title with icon
st.markdown('<h1>🧠 MEG Coregistration Visualization</h1>', unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("""
    <div style='text-align: center; padding: 0px;'>
        <h2 >⚙️ Settings</h2>
    </div>
""", unsafe_allow_html=True)

if in_docker():
    report_root_dir = Path("/output")
    default_subjects_dir = Path("/smri")
else:
    report_root_dir = Path(st.session_state.get("dataset_report_path"))
    default_subjects_dir = Path(st.session_state.get("subjects_dir"))

default_meg_dir = report_root_dir / "preprocessed"
default_trans_dir = report_root_dir / "preprocessed" / "trans"

# Directory Configuration
st.sidebar.markdown("#### 📁 Directory Paths")
subjects_dir = st.sidebar.text_input("FreeSurfer SUBJECTS_DIR", default_subjects_dir)

# Subject Selection
if os.path.exists(subjects_dir):
    subjects = sorted([
        f for f in os.listdir(subjects_dir)
        if os.path.isdir(os.path.join(subjects_dir, f)) and not f.startswith('fsaverage')
    ])
    selected_subject = st.sidebar.selectbox("Select Subject", subjects)
    st.sidebar.markdown(f'<div class="success-box">✅ Found {len(subjects)} subjects</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="warning-box">⚠️ No valid SUBJECTS_DIR found at: {subjects_dir}</div>',
                        unsafe_allow_html=True)
    selected_subject = None

# MEG File Selection
pattern = os.path.join(default_meg_dir, f"{selected_subject}*")
matched_dirs = glob.glob(pattern)
if matched_dirs:
    default_meg_dir = matched_dirs[0]

meg_dir = st.sidebar.text_input("MEG Directory", default_meg_dir)

if os.path.exists(meg_dir):
    meg_files = []
    for root, dirs, files in os.walk(meg_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in ['.fif', '.ds', '.sqd', '.con']):
                rel_path = os.path.relpath(os.path.join(root, f), meg_dir)
                meg_files.append(rel_path)
    meg_files = sorted(meg_files)

    if meg_files:
        selected_meg_file = st.sidebar.selectbox("📊 Select MEG File", meg_files)
        selected_meg_file = os.path.join(meg_dir, selected_meg_file)
        st.sidebar.markdown(f'<div class="success-box">✅ Found {len(meg_files)} MEG files</div>',
                            unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="warning-box">⚠️ No MEG files found</div>', unsafe_allow_html=True)
        selected_meg_file = None
else:
    st.sidebar.markdown(f'<div class="warning-box">⚠️ MEG directory not found for subject: {selected_subject}</div>',
                        unsafe_allow_html=True)
    meg_files = []
    selected_meg_file = None

# Transform File Selection
trans_dir = st.sidebar.text_input("Transform Directory", default_trans_dir)
trans_dirs = sorted([f for f in os.listdir(trans_dir)])
selected_trans = st.sidebar.selectbox("🔄 Select Transform File", trans_dirs)
selected_trans_file = Path(trans_dir) / selected_trans / "coreg-trans.fif"

st.sidebar.markdown("---")

# Visualization Settings
st.sidebar.markdown('⚙️ Visualization Settings', unsafe_allow_html=True)
opacity = st.sidebar.slider("Scalp Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

st.sidebar.markdown("---")

# Info Display
print("Selected MEG:", selected_meg_file)
print("Selected Subject:", selected_subject)
print("Selected Transform:", selected_trans_file)

# Display current configuration in main area
col1, col2= st.columns(2)

with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Subject</div><div class="metric-value">👤 ' + str(
        selected_subject) + '</div></div>', unsafe_allow_html=True)

with col2:
    meg_filename = Path(selected_meg_file).name if selected_meg_file else "N/A"
    st.markdown(
        '<div class="metric-card"><div class="metric-title">MEG File</div><div class="metric-value">📊 ' + meg_filename + '</div></div>',
        unsafe_allow_html=True)



st.markdown("---")

# Legend for fiducial points
st.markdown('<div class="section-header">📍 Fiducial Point Legend</div>', unsafe_allow_html=True)

legend_html = """
<div style="padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem;">
    <div class="legend-item">
        <span class="legend-color" style="background-color: orange;"></span>
        <span class="legend-text">Nasion</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background-color: blue;"></span>
        <span class="legend-text">LPA (Left)</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background-color: red;"></span>
        <span class="legend-text">RPA (Right)</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background-color: purple;"></span>
        <span class="legend-text">HPI Points</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background-color: salmon;"></span>
        <span class="legend-text">HSP Points</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background-color: gray;"></span>
        <span class="legend-text">Brain/Scalp</span>
    </div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# Load data and create visualization
try:
    with st.spinner('🔄 Loading data and generating visualization...'):
        t1_mgh = load_t1(subject=selected_subject, subjects_dir=subjects_dir)
        raw = mne.io.read_raw(selected_meg_file)
        coreg_trans = mne.read_trans(selected_trans_file)

        print("coreg_trans", coreg_trans)
        print("coreg_trans matrix:", coreg_trans['trans'])

        plotter = pv.Plotter(window_size=[800, 600])
        plotter = visualize_head_surface(
            subject=selected_subject,
            subjects_dir=subjects_dir,
            trans=coreg_trans,
            raw=raw,
            t1_mgh=t1_mgh,
            opacity=opacity
        )

        plotter = visualize_nasion_and_scalp(
            plotter=plotter,
            subject=selected_subject,
            subjects_dir=subjects_dir,
            trans=coreg_trans,
            raw=raw,
            t1_mgh=t1_mgh
        )

        plotter.set_background("white")
        plotter.view_isometric()
        plotter.add_axes_at_origin()

    # Display visualization
    st.markdown('<div class="section-header">🎯 3D Visualization</div>', unsafe_allow_html=True)
    stpyvista(
        plotter,
        key=f"coregistration_{time.time()}",
        panel_kwargs=dict(interactive_orientation_widget=False, orientation_widget=True)
    )

    # Display transformation matrix
    st.markdown("---")
    st.markdown('<div class="section-header">🔢 Coregistration Transform Matrix (4×4)</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">This homogeneous transformation matrix represents the spatial relationship between MRI and MEG head coordinates.</div>',
        unsafe_allow_html=True)

    # Format the matrix nicely
    transform_df = coreg_trans['trans']
    st.dataframe(
        transform_df,
        use_container_width=True,
        height=200
    )

    # Additional matrix info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Matrix Shape", f"{transform_df.shape[0]} × {transform_df.shape[1]}")
    with col2:
        determinant = np.linalg.det(transform_df[:3, :3])
        st.metric("Rotation Determinant", f"{determinant:.6f}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #6c757d; padding: 1rem;">'
    'MEG Coregistration Visualization Tool | Built with Streamlit & PyVista'
    '</div>',
    unsafe_allow_html=True
)
