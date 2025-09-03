# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import pyvista as pv
import mne
import altair as alt
from pathlib import Path

# --- 1. 设置环境变量 (在无头环境中避免 GLSL 错误) ---
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"

# 如果在无头环境，需要指定 DISPLAY
# virtual GUI |
# Xvfb :99 -screen 0 1920x1080x24 &
os.environ["DISPLAY"] = ":99"

# --- 2. 启动虚拟显示 (Xvfb) ---
if "IS_XVFB_RUNNING" not in st.session_state:
    start_xvfb()
    st.session_state.IS_XVFB_RUNNING = True

# --- 3. 页面布局 ---
st.title("MNE Source Localization")
st.write("### Surface View (Left & Right Hemispheres) + RMS Time Series")

# --- 4. 缓存函数以加载 SourceEstimate 数据 ---
# @st.cache_data(show_spinner=False)
def load_stc(subject):
    """
    加载左右半球的 SourceEstimate 数据。

    Parameters:
    - subject: str, 被试名称

    Returns:
    - stc_lh: 左半球的 SourceEstimate 对象
    - stc_rh: 右半球的 SourceEstimate 对象
    """
    # stc_file_lh = mne.datasets.sample.data_path() / "MEG" / "sample" / "sample_audvis-meg-eeg-lh.stc"
    # stc_file_rh = mne.datasets.sample.data_path() / "MEG" / "sample" / "sample_audvis-meg-eeg-rh.stc"
    stc_file_lh = mne.datasets.sample.data_path() / "MEG" / "sample" / "sample_audvis-meg-lh.stc"
    stc_file_rh = mne.datasets.sample.data_path() / "MEG" / "sample" / "sample_audvis-meg-rh.stc"

    stc_lh = mne.read_source_estimate(stc_file_lh, subject=subject)
    stc_rh = mne.read_source_estimate(stc_file_rh, subject=subject)

    return stc_lh, stc_rh


# --- 5. 缓存函数以加载脑表面数据 ---
# @st.cache_data(show_spinner=False)
def load_surface(subject, surf='white'):
    """
    加载左右半球的脑表面数据。

    Parameters:
    - subject: str, 被试名称
    - surf: str, 表面类型（'white', 'pial', 'inflated' 等）

    Returns:
    - meshes: dict, 包含左右半球的 PyVista PolyData 对象
    """
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    hemispheres = ['lh', 'rh']
    meshes = {}

    for hemi in hemispheres:
        surface_path = subjects_dir / subject / "surf" / f"{hemi}.{surf}"
        coords, faces = mne.read_surface(surface_path)
        # PyVista 需要 faces 数组的前面多一个多边形的顶点数量（3）
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        mesh = pv.PolyData(coords, faces)
        meshes[hemi] = mesh
    return meshes


# --- 6. 加载数据 ---
# subject = "fsaverage"
# surf = "white"  # 可选：'white', 'pial', 'inflated'
#
# # 加载 SourceEstimate 数据
# stc_lh, stc_rh = load_stc(subject)
#



###############################################################################
# 6) 主逻辑：加载 stc_lh, stc_rh，并在 MNE 中 morph 平滑
###############################################################################
subject = "fsaverage"  # 与您原代码一致
surf = "white"

# 先加载 (单半球) stc_lh, stc_rh
stc_lh, stc_rh = load_stc(subject)

# 提供一个滑块来控制平滑迭代次数
smooth_iter = st.sidebar.slider("Smoothing iterations (smooth )", min_value=0, max_value=200, value=1)

# st.write("对单半球 stc 分别执行 morph(smooth=...)，以实现基于脑表面拓扑的平滑。")

subjects_dir = mne.datasets.sample.data_path() / "subjects"

# 3) 对 stc_lh 做 morph

morph_lh = mne.compute_source_morph(
    src=stc_lh,
    subject_from=subject,
    subject_to=subject,     # 只在同一个subject上平滑
    subjects_dir=subjects_dir,
    smooth=smooth_iter
)
stc_lh = morph_lh.apply(stc_lh)

# 4) 对 stc_rh 做 morph
morph_rh = mne.compute_source_morph(
    src=stc_rh,
    subject_from=subject,
    subject_to=subject,
    subjects_dir=subjects_dir,
    smooth=smooth_iter
)
stc_rh = morph_rh.apply(stc_rh)

# st.write("Morph Done.")


# --- 7. 预先计算 RMS 时间序列 ---
# 简单示例：把左、右半球数据拼起来, 做 RMS(t) = sqrt( mean( data^2 ) ) over all vertices
# data_lh.shape = (#lh_vertices, #times)
# data_rh.shape = (#rh_vertices, #times)
all_data = np.concatenate([stc_lh.data, stc_rh.data], axis=0)  # shape=(n_lh + n_rh, n_times)
# RMS over vertices:
rms_all = np.sqrt(np.mean(all_data ** 2, axis=0))  # shape=(n_times, )

# 创建 DataFrame 用于绘图
df_rms = pd.DataFrame({
    "time (s)": stc_lh.times,
    "RMS": rms_all
})

# --- 8. 添加时间滑块 ---
time_min = float(stc_lh.times[0])
time_max = float(stc_lh.times[-1])
time_init = float(stc_lh.times[0])

time_point = st.slider(
    "**Select time point (seconds)**",
    min_value=time_min,
    max_value=time_max,
    value=time_init,
    step=0.01
)

# 将时间点转换为最近的索引 (lh 与 rh 时间点应相同)
time_idx = np.argmin(np.abs(stc_lh.times - time_point))

# 获取当前 RMS 值
rms_current = rms_all[time_idx]


# --- 9. 映射 SourceEstimate 数据到网格 (lh & rh) ---
def map_stc_to_mesh(stc, mesh, hemi, time_index):
    """
    将 SourceEstimate 数据映射到 PyVista 网格的顶点上。

    Parameters:
    - stc: SourceEstimate 对象
    - mesh: PyVista PolyData 对象
    - hemi: 'lh' 或 'rh'
    - time_index: int, 时间点索引

    Returns:
    - scalars: np.ndarray, 映射后的标量数组
    """
    # print("Data min =", stc.data[:, time_idx].min())
    # print("Data max =", stc.data[:, time_idx].max())
    #
    # print("GData min =", stc.data.min())
    # print("GData max =", stc.data.max())
    # print(np.histogram(stc.data,bins=100))

    print("stc.data",stc.data.shape)
    if hemi == 'lh':
        vertno = stc.lh_vertno
        print("lh vertno:",vertno,len(vertno))
        data = stc.data[:len(vertno), time_index]
    elif hemi == 'rh':
        vertno = stc.rh_vertno
        print("rh vertno:", vertno, len(vertno))
        data = stc.data[:len(vertno), time_index]
    else:
        raise ValueError("hemi must be 'lh' or 'rh'")

    scalars = np.zeros(mesh.n_points)
    # scalars = np.full(mesh.n_points, -100, dtype=np.float64)
    # 使用 NumPy 的高级索引进行快速赋值
    scalars[vertno] = data
    return scalars

alldata = np.concatenate([stc_lh.data, stc_rh.data], axis=0)
global_min, global_max = alldata.min(), alldata.max()

# # 加载脑表面数据
meshes = load_surface(subject, surf)

# 映射数据到左半球和右半球
scalars_lh = map_stc_to_mesh(stc_lh, meshes['lh'], 'lh', time_idx)
scalars_rh = map_stc_to_mesh(stc_rh, meshes['rh'], 'rh', time_idx)
print("scalars_lh:",scalars_lh,scalars_lh.shape,np.max(scalars_lh),np.min(scalars_lh))

# 将标量数据添加到网格
meshes['lh']["source_activation"] = scalars_lh
meshes['rh']["source_activation"] = scalars_rh

print("np.min(scalars_lh), np.max(scalars_lh):",np.min(scalars_lh), np.max(scalars_lh))
# --- 10. 管理 PyVista Plotter ---
if "plotter" not in st.session_state:
    # 第一次运行时，创建新的 Plotter 并添加网格
    plotter = pv.Plotter(window_size=[800, 600])

    # 添加左半球 bwr
    plotter.add_mesh(
        meshes['lh'],
        scalars="source_activation",
        cmap="bwr",
        clim=[global_min, global_max],
        show_edges=False,
        opacity=1,
        name="lh_mesh"
    )

    # 添加右半球
    plotter.add_mesh(
        meshes['rh'],
        scalars="source_activation",
        cmap="bwr",
        clim=[global_min, global_max],
        show_edges=False,
        opacity=1,
        name="rh_mesh"
    )

    # 添加标量条
    plotter.add_scalar_bar(
        title="Source Activation",
        label_font_size=10,
        title_font_size=12,
    )

    # 设置背景 & 视角
    plotter.set_background("white")
    plotter.view_isometric()

    st.session_state.plotter = plotter
else:
    # 后续运行时，获取已存在的 Plotter 并更新网格
    plotter = st.session_state.plotter

    # 清理已有内容
    plotter.clear()

    # 重新添加左半球
    plotter.add_mesh(
        meshes['lh'],
        scalars="source_activation",
        cmap="bwr",
        clim=[global_min, global_max],
        show_edges=False,
        opacity=1,
        name="lh_mesh"
    )

    # 重新添加右半球
    plotter.add_mesh(
        meshes['rh'],
        scalars="source_activation",
        cmap="bwr",
        clim=[global_min, global_max],
        show_edges=False,
        opacity=1,
        name="rh_mesh"
    )

    # 重新添加标量条
    plotter.add_scalar_bar(
        title="Source Activation",
        label_font_size=10,
        title_font_size=12,
    )

    # 设置背景 & 视角
    plotter.set_background("white")
    plotter.view_isometric()

# --- 11. 在 Streamlit 中渲染 Plotter ---
stpyvista(plotter, key="brain_plot")


# --- 13. 显示 RMS 曲线图 + 当前时间点 RMS 值 ---
st.write("### RMS Over Time")

# 创建 Altair 图表
base = alt.Chart(df_rms).mark_line(color='blue').encode(
    x=alt.X('time (s)', title='Time (s)'),
    y=alt.Y('RMS', title='RMS')
)

# 创建竖线 (Rule) 标记当前时间点
rule = alt.Chart(pd.DataFrame({'time (s)': [time_point]})).mark_rule(color='red').encode(
    x='time (s)'
)

# 组合图表
chart = base + rule

# 渲染图表
st.altair_chart(chart, use_container_width=True)

# 显示当前时间点的 RMS 值
st.write(f"**Current Time**: {time_point:.3f} s， **RMS**: {rms_current:.4f}")

# --- 12. 添加“关闭 Plotter”按钮 ---
if st.button("Close Plotter"):
    plotter.close()
    del st.session_state.plotter
    st.success("Plotter Closed! Please Refresh.")
