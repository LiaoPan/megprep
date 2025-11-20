#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import pyvista as pv
import mne
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# --- 1. 设置环境变量 ---
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
os.environ["DISPLAY"] = ":99"

# --- 2. 启动虚拟显示 ---
if "IS_XVFB_RUNNING" not in st.session_state:
    start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True


# --- 3. 阈值处理函数 ---
def apply_threshold_to_data(data, threshold, transparent_value=0.0):
    """对数据应用阈值，低于阈值的设为透明值"""
    data_thresholded = data.copy()
    mask = np.abs(data) < threshold
    data_thresholded[mask] = transparent_value
    return data_thresholded, mask


# --- 4. 自定义 CSS 样式 ---
st.markdown("""
    <style>
        .main-header {
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-weight: 700;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h2 class="main-header">🧠 Source Localization 3D Viewer</h2>', unsafe_allow_html=True)

# --- 5. 侧边栏配置 ---
st.sidebar.header("⚙️ Visualization Settings")

# 预设模式
st.sidebar.subheader("🎨 Visualization Presets")
preset = st.sidebar.selectbox(
    "Choose preset",
    [
        "Custom",
        "High Contrast (95-99.9%)",
        "Very High Contrast (97-99.95%)",
        "Extreme Contrast (98-99.99%)",
        "Default",
    ],
    index=4,
    help="Presets for common visualization styles"
)

# 预设配置
preset_configs = {
    "High Contrast (95-99.9%)": {
        "threshold_pct": 95.0,
        "ctrl_pts": [95.0, 98.0, 99.9],
        "gray_opacity": 0.7,
    },
    "Very High Contrast (97-99.95%)": {
        "threshold_pct": 97.0,
        "ctrl_pts": [97.0, 99.0, 99.95],
        "gray_opacity": 0.75,
    },
    "Extreme Contrast (98-99.99%)": {
        "threshold_pct": 98.0,
        "ctrl_pts": [98.0, 99.5, 99.99],
        "gray_opacity": 0.8,
    },
    "Default": {
        "threshold_pct": 60.0,
        "ctrl_pts": [90.0, 97.0, 99.95],
        "gray_opacity": 0.6,
    },
}

# 数据路径配置
with st.sidebar.expander("📁 Data Paths", expanded=False):
    subject = st.text_input("Subject ID", value="sub-01")
    subjects_dir = st.text_input("Subjects Directory",
                                 value="/data/liaopan/datasets/SMN4Lang_smri/")
    stc_lh_path = st.text_input("Left Hemisphere STC",
                                value="/data/liaopan/datasets/SMN4Lang/g_nx/preprocessed/source_recon/sub-01_task-RDR_run-1_meg/wdonset_evoked_dSPM-ico4-lh.stc")
    stc_rh_path = st.text_input("Right Hemisphere STC",
                                value="/data/liaopan/datasets/SMN4Lang/g_nx/preprocessed/source_recon/sub-01_task-RDR_run-1_meg/wdonset_evoked_dSPM-ico4-rh.stc")

subjects_dir = Path(subjects_dir)

# 表面和平滑参数
st.sidebar.subheader("🎨 Surface Parameters")
surf = st.sidebar.selectbox("Surface Type", ["inflated", "white", "pial"], index=0)
spacing = st.sidebar.slider("ICO Spacing", min_value=3, max_value=5, value=4)
smooth_iter = st.sidebar.slider("Smoothing Iterations", min_value=0, max_value=200, value=10)

# 半球分离控制
st.sidebar.subheader("🔀 Hemisphere Separation")
separate_hemis = st.sidebar.checkbox(
    "Separate hemispheres",
    value=True,
    help="Separate left and right hemispheres for better visualization"
)

if separate_hemis:
    separation_distance = st.sidebar.slider(
        "Separation distance (mm)",
        min_value=0.0,
        max_value=50.0,
        value=50.0,
        step=1.0,
        help="Distance between hemispheres along X-axis"
    )
else:
    separation_distance = 0.0

# 半球选择
st.sidebar.subheader("🧠 Hemisphere View")
hemisphere_view = st.sidebar.selectbox(
    "Display",
    options=["Both", "Left Only", "Right Only"],
    index=0
)

# 可视化参数
st.sidebar.subheader("🎨 Visualization Parameters")
opacity = st.sidebar.slider("Brain opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

# 颜色图选择
st.sidebar.markdown("**Colormap Selection**")
cmap_type = st.sidebar.radio(
    "Colormap Type",
    ["Diverging (±)", "Sequential (0→)"],
    index=0
)

if cmap_type == "Diverging (±)":
    colormap = st.sidebar.selectbox(
        "Diverging Colormap",
        ["bwr", "coolwarm", "RdBu_r", "seismic", "PiYG"],
        index=0
    )
    default_symmetric = True
else:
    colormap = st.sidebar.selectbox(
        "Sequential Colormap",
        ["hot", "Reds", "YlOrRd", "OrRd", "Oranges"],
        index=0
    )
    default_symmetric = False

# 阈值控制
st.sidebar.subheader("🎯 Threshold Control")
use_threshold = st.sidebar.checkbox("Apply threshold", value=True,
                                    help="Hide low activation areas (like MNE)")

if use_threshold:
    threshold_mode = st.sidebar.radio(
        "Threshold mode",
        ["Percentile", "Absolute value"],
        index=0
    )

    if preset != "Custom" and preset in preset_configs:
        default_threshold = preset_configs[preset]["threshold_pct"]
    else:
        default_threshold = 60.0

    if threshold_mode == "Percentile":
        threshold_pct = st.sidebar.slider(
            "Threshold percentile (%)",
            0.0, 100.0, default_threshold,
            0.1,
            help="Only show values above this percentile"
        )
    else:
        threshold_val = st.sidebar.number_input(
            "Threshold value",
            0.0, None, 10.0,
            0.5,
            help="Hide absolute values below this"
        )

# 灰色区域控制
if preset != "Custom" and preset in preset_configs:
    default_gray = preset_configs[preset]["gray_opacity"]
else:
    default_gray = 0.75

gray_opacity = st.sidebar.slider(
    "Gray area opacity",
    min_value=0.0,
    max_value=1.0,
    value=default_gray,
    step=0.05,
    help="Opacity of low-activation (gray) areas"
)

# 🆕 灰色亮度控制
gray_brightness = st.sidebar.slider(
    "Gray brightness",
    min_value=0.3,
    max_value=0.9,
    value=0.65,
    step=0.05,
    help="Brightness of gray areas (higher = lighter)"
)

# 颜色范围控制
st.sidebar.subheader("📊 Color Limit Control")

clim_mode = st.sidebar.radio(
    "Control Mode",
    ["Percentile", "Absolute Value"],
    index=0
)

use_symmetric = st.sidebar.checkbox(
    "Symmetric limits (pos_lims)",
    value=default_symmetric,
    help="For diverging colormaps"
)

st.sidebar.markdown("**Control Points** (lower, middle, upper)")

if preset != "Custom" and preset in preset_configs:
    default_pts = preset_configs[preset]["ctrl_pts"]
else:
    default_pts = [97.0, 99.0, 99.95]

if clim_mode == "Percentile":
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        p1 = st.number_input("Lower %", 0.0, 100.0, default_pts[0], 0.1, format="%.1f", key="p1")
    with col2:
        p2 = st.number_input("Mid %", 0.0, 100.0, default_pts[1], 0.1, format="%.1f", key="p2")
    with col3:
        p3 = st.number_input("Upper %", 0.0, 100.0, default_pts[2], 0.01, format="%.2f", key="p3")

    ctrl_pts = [p1, p2, p3]
    clim_kind = "percent"
else:
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        v1 = st.number_input("Lower", value=8.0, format="%.2f", key="v1")
    with col2:
        v2 = st.number_input("Mid", value=12.0, format="%.2f", key="v2")
    with col3:
        v3 = st.number_input("Upper", value=18.0, format="%.2f", key="v3")

    ctrl_pts = [v1, v2, v3]
    clim_kind = "value"


# --- 6. 数据加载函数 ---
@st.cache_data(show_spinner=False)
def load_stc(stc_lh_path, stc_rh_path, subject):
    try:
        stc_lh = mne.read_source_estimate(stc_lh_path, subject=subject)
        stc_rh = mne.read_source_estimate(stc_rh_path, subject=subject)
        return stc_lh, stc_rh, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(show_spinner=False)
def load_surface(_subjects_dir, subject, surf='white'):
    hemispheres = ['lh', 'rh']
    meshes = {}
    try:
        for hemi in hemispheres:
            surface_path = _subjects_dir / subject / "surf" / f"{hemi}.{surf}"
            coords, faces = mne.read_surface(surface_path)
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            mesh = pv.PolyData(coords, faces)
            meshes[hemi] = mesh
        return meshes, None
    except Exception as e:
        return None, str(e)


def apply_morphing(stc, subject, subjects_dir, smooth_iter, spacing):
    morph = mne.compute_source_morph(
        src=stc,
        subject_from=subject,
        subject_to=subject,
        subjects_dir=subjects_dir,
        smooth=smooth_iter,
        spacing=spacing
    )
    return morph.apply(stc)


def map_stc_to_mesh_interpolated(stc, mesh, hemi, time_index):
    """使用 KD-Tree 插值将 SourceEstimate 数据映射到完整表面网格"""
    if hemi == 'lh':
        vertno = stc.lh_vertno
        data = stc.data[:len(vertno), time_index]
    elif hemi == 'rh':
        vertno = stc.rh_vertno
        data = stc.data[:len(vertno), time_index]
    else:
        raise ValueError("hemi must be 'lh' or 'rh'")

    src_coords = mesh.points[vertno]
    target_coords = mesh.points

    tree = cKDTree(src_coords)
    distances, indices = tree.query(target_coords, k=12) # 2 --> 12

    weights = 1.0 / (distances + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)

    scalars = np.sum(data[indices] * weights, axis=1)

    return scalars





# --- 7. 主程序流程 ---
with st.spinner("🔄 Loading source estimate data..."):
    stc_lh, stc_rh, error = load_stc(stc_lh_path, stc_rh_path, subject)

if error:
    st.error(f"❌ Error loading STC data: {error}")
    st.stop()

with st.spinner("🔄 Applying surface-based smoothing..."):
    stc_lh = apply_morphing(stc_lh, subject, subjects_dir, smooth_iter, spacing)
    stc_rh = apply_morphing(stc_rh, subject, subjects_dir, smooth_iter, spacing)

with st.spinner("🔄 Loading brain surface meshes..."):
    meshes_original, error = load_surface(subjects_dir, subject, surf)

if error:
    st.error(f"❌ Error loading surface: {error}")
    st.stop()

# --- 8. 准备数据 ---
if hemisphere_view == "Left Only":
    alldata = stc_lh.data
elif hemisphere_view == "Right Only":
    alldata = stc_rh.data
else:
    alldata = np.concatenate([stc_lh.data, stc_rh.data], axis=0)

# --- 9. 计算颜色范围 ---
if clim_kind == "percent":
    if use_symmetric:
        perc_data = np.abs(alldata)
        fmin, fmid, fmax = np.percentile(perc_data, ctrl_pts)
    else:
        fmin, fmid, fmax = np.percentile(alldata, ctrl_pts)
else:
    fmin, fmid, fmax = ctrl_pts

if use_symmetric:
    clim = (-fmax, fmax)
    display_fmin = fmin
else:
    clim = (fmin, fmax)
    display_fmin = fmin

# 计算阈值
if use_threshold:
    if threshold_mode == "Percentile":
        threshold = np.percentile(np.abs(alldata), threshold_pct)
    else:
        threshold = threshold_val
else:
    threshold = 0.0

if preset != "Custom":
    st.sidebar.success(f"📋 Using preset: **{preset}**")

st.sidebar.info(f"✅ Threshold: **{threshold:.4f}**")
st.sidebar.info(f"✅ Color range: **[{clim[0]:.4f}, {clim[1]:.4f}]**")

# 计算 RMS
rms_all = np.sqrt(np.mean(alldata ** 2, axis=0))

# --- 10. 显示数据信息 ---
col1, col2, col3, col4 = st.columns([1,1,1,2])
with col1:
    if hemisphere_view in ["Both", "Left Only"]:
        st.metric("📊 Vertices (LH)", f"{len(stc_lh.lh_vertno):,}")
    else:
        st.metric("📊 Vertices (LH)", "—")
with col2:
    if hemisphere_view in ["Both", "Right Only"]:
        st.metric("📊 Vertices (RH)", f"{len(stc_rh.rh_vertno):,}")
    else:
        st.metric("📊 Vertices (RH)", "—")
with col3:
    st.metric("⏱️ Time Points", f"{len(stc_lh.times):,}")
with col4:
    st.metric("📏 Time Range", f"{stc_lh.times[0]:.2f} ~ {stc_lh.times[-1]:.2f} s")

# --- 11. 时间选择器 ---
st.markdown("---")
st.subheader("⏰ Time Point Selection")

time_min = float(stc_lh.times[0])
time_max = float(stc_lh.times[-1])
time_init = min(0.31, time_max)

time_point = st.slider(
    "Select time point (seconds)",
    min_value=time_min,
    max_value=time_max,
    value=time_init,
    step=0.001,
    format="%.3f"
)

time_idx = np.argmin(np.abs(stc_lh.times - time_point))
actual_time = stc_lh.times[time_idx]
rms_current = rms_all[time_idx]

# --- 12. RMS 时间序列图 ---
st.markdown("---")
st.subheader(f"📊 Root Mean Square (RMS) - {hemisphere_view}")

fig_rms, ax = plt.subplots(figsize=(12, 4))
ax.plot(stc_lh.times, rms_all, color='#2E86AB', linewidth=2, label='RMS')
ax.fill_between(stc_lh.times, rms_all, alpha=0.2, color='#2E86AB')
ax.axvline(actual_time, color='#E63946', linestyle='--', linewidth=2.5, label='Current Time')
ax.plot(actual_time, rms_current, 'o', color='#E63946', markersize=10,
        markeredgecolor='white', markeredgewidth=2, zorder=5)

if use_threshold:
    ax.axhline(threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold: {threshold:.3f}')

ax.axhline(fmin, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'fmin: {fmin:.3f}')

ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMS Activation', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()

st.pyplot(fig_rms)
plt.close()

# --- 13. 显示当前统计信息 ---
st.markdown("### 📈 Current Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; color: white; box-shadow: 0 6px 15px rgba(0,0,0,0.1);'>
        <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>⏰ Time</div>
        <div style='font-size: 36px; font-weight: bold;'>{actual_time:.3f}</div>
        <div style='font-size: 14px; opacity: 0.85; margin-top: 5px;'>seconds</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                border-radius: 12px; color: white; box-shadow: 0 6px 15px rgba(0,0,0,0.1);'>
        <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>📊 RMS</div>
        <div style='font-size: 36px; font-weight: bold;'>{rms_current:.4f}</div>
        <div style='font-size: 14px; opacity: 0.85; margin-top: 5px;'>activation</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                border-radius: 12px; color: white; box-shadow: 0 6px 15px rgba(0,0,0,0.1);'>
        <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>🎯 Threshold</div>
        <div style='font-size: 36px; font-weight: bold;'>{threshold:.3f}</div>
        <div style='font-size: 14px; opacity: 0.85; margin-top: 5px;'>cutoff</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                border-radius: 12px; color: white; box-shadow: 0 6px 15px rgba(0,0,0,0.1);'>
        <div style='font-size: 16px; opacity: 0.9; margin-bottom: 8px;'>🎨 Range</div>
        <div style='font-size: 36px; font-weight: bold;'>±{clim[1]:.2f}</div>
        <div style='font-size: 14px; opacity: 0.85; margin-top: 5px;'>color limit</div>
    </div>
    """, unsafe_allow_html=True)

# --- 14. 创建脑表面可视化 ---
st.markdown("---")
st.subheader(f"🧠 Brain Surface Visualization - {hemisphere_view}")

with st.spinner("Rendering brain surface..."):
    plotter = pv.Plotter(window_size=[1400, 700])

    n_vertices_shown = 0
    n_vertices_hidden = 0

    # 🔑 计算灰色RGB值
    gray_rgb_val = int(gray_brightness * 255)
    gray_color = f"#{gray_rgb_val:02x}{gray_rgb_val:02x}{gray_rgb_val:02x}"

    if hemisphere_view in ["Both", "Left Only"]:
        scalars_lh = map_stc_to_mesh_interpolated(stc_lh, meshes_original['lh'], 'lh', time_idx)

        if use_threshold:
            scalars_lh_thresholded, mask_lh = apply_threshold_to_data(
                scalars_lh, threshold, transparent_value=np.nan
            )
            n_vertices_hidden += mask_lh.sum()
            n_vertices_shown += (~mask_lh).sum()
        else:
            scalars_lh_thresholded = scalars_lh
            n_vertices_shown += len(scalars_lh)

        mesh_lh_display = meshes_original['lh'].copy()
        if separate_hemis:
            mesh_lh_display.points[:, 0] -= separation_distance

        mesh_lh_display["activation"] = scalars_lh_thresholded

        # 🆕 改进的渲染参数
        plotter.add_mesh(
            mesh_lh_display,
            scalars="activation",
            cmap=colormap,
            clim=clim,
            show_edges=False,
            opacity=opacity,
            smooth_shading=True,
            nan_color=gray_color,  # 使用可调节的灰色
            nan_opacity=gray_opacity,
            # ambient=0.3,  # 🆕 环境光
            # diffuse=0.6,  # 🆕 漫反射
            # specular=0.3,  # 🆕 镜面反射
            # specular_power=20,  # 🆕 镜面反射强度
            name="lh_mesh"
        )

    if hemisphere_view in ["Both", "Right Only"]:
        scalars_rh = map_stc_to_mesh_interpolated(stc_rh, meshes_original['rh'], 'rh', time_idx)

        if use_threshold:
            scalars_rh_thresholded, mask_rh = apply_threshold_to_data(
                scalars_rh, threshold, transparent_value=np.nan
            )
            n_vertices_hidden += mask_rh.sum()
            n_vertices_shown += (~mask_rh).sum()
        else:
            scalars_rh_thresholded = scalars_rh
            n_vertices_shown += len(scalars_rh)

        mesh_rh_display = meshes_original['rh'].copy()
        if separate_hemis:
            mesh_rh_display.points[:, 0] += separation_distance

        mesh_rh_display["activation"] = scalars_rh_thresholded

        plotter.add_mesh(
            mesh_rh_display,
            scalars="activation",
            cmap=colormap,
            clim=clim,
            show_edges=False,
            opacity=opacity,
            smooth_shading=True,
            nan_color=gray_color,
            nan_opacity=gray_opacity,
            # ambient=0.3,
            # diffuse=0.6,
            # specular=0.3,
            # specular_power=20,
            name="rh_mesh"
        )

    plotter.add_scalar_bar(
        title=f"Activation",
        label_font_size=14,
        title_font_size=16,
        n_labels=5,
        italic=False,
        bold=True,
        shadow=False,
        color='black',
        position_x=0.82,
        position_y=0.05,
        width=0.15,
        height=0.6
    )

    # 🆕 改进的光照设置
    plotter.set_background("white")
    plotter.enable_anti_aliasing('fxaa')  # 抗锯齿

    # 设置视角
    if hemisphere_view == "Left Only":
        plotter.camera_position = 'xy'
        plotter.camera.azimuth = -90
        plotter.camera.elevation = 10
    elif hemisphere_view == "Right Only":
        plotter.camera_position = 'xy'
        plotter.camera.azimuth = 90
        plotter.camera.elevation = 10
    else:
        if separate_hemis:
            plotter.camera_position = 'xy'
            plotter.camera.elevation = 15
            plotter.camera.azimuth = 0
        else:
            plotter.view_isometric()

    # 渲染
    render_key = f"brain_{hemisphere_view}_{time_idx}_{colormap}_{threshold:.4f}_{clim[0]:.4f}_{clim[1]:.4f}_{separation_distance:.1f}_{gray_opacity:.2f}_{gray_brightness:.2f}_{preset}_{time.time()}"
    stpyvista(plotter, key=render_key)

# 显示统计
percentage_shown = (n_vertices_shown / (n_vertices_shown + n_vertices_hidden) * 100) if (
                                                                                                    n_vertices_shown + n_vertices_hidden) > 0 else 0

if use_threshold:
    st.info(
        f"🎯 **Activation Coverage**: {percentage_shown:.2f}% of brain surface ({n_vertices_shown:,} / {n_vertices_shown + n_vertices_hidden:,} vertices)")

if separate_hemis:
    st.success(f"🔀 **Hemispheres separated** by {separation_distance:.1f} mm along X-axis")

st.info(f"🎨 **Gray color**: {gray_color} (brightness: {gray_brightness:.2f})")

# --- 15. 详细信息 ---
with st.expander("📊 Data Statistics & Configuration"):
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Data Statistics ({hemisphere_view}):**")
        st.write(f"- Shape: `{alldata.shape}`")
        st.write(f"- Min: `{alldata.min():.6f}`")
        st.write(f"- Max: `{alldata.max():.6f}`")
        st.write(f"- Mean: `{alldata.mean():.6f}`")
        st.write(f"- Std: `{alldata.std():.6f}`")
        st.write(f"- Abs Max: `{np.abs(alldata).max():.6f}`")

        st.write(f"\n**Key Percentiles (Absolute Values):**")
        for pct in [90, 95, 96, 97, 98, 99, 99.5, 99.9, 99.95]:
            val = np.percentile(np.abs(alldata), pct)
            marker = "👉" if abs(pct - threshold_pct) < 0.1 else "  "
            st.write(f"{marker} {pct:5.2f}%: `{val:.6f}`")

    with col2:
        st.write("**Visualization Configuration:**")
        st.json({
            "preset": preset,
            "colormap": colormap,
            "gray_color": gray_color,
            "gray_brightness": float(gray_brightness),
            "gray_opacity": float(gray_opacity),
            "clim": [float(clim[0]), float(clim[1])],
            "threshold": {
                "value": float(threshold),
                "percentile": float(threshold_pct) if use_threshold and threshold_mode == "Percentile" else None
            },
            "lighting": {
                "ambient": 0.3,
                "diffuse": 0.6,
                "specular": 0.3
            }
        })

st.markdown("---")
st.markdown("""
### 💡 Tips for Optimal Visualization:
1. **Gray brightness**: 0.6-0.7 for good contrast (currently showing as hex color above)
2. **Gray opacity**: 0.7-0.8 to see brain structure clearly
3. **Threshold**: 97-98% for publication quality
4. **If brain looks too dark**: Increase gray brightness
5. **If brain looks washed out**: Decrease gray brightness
""")