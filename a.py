#         Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

import mne
from mne.coreg import Coregistration
from mne.io import read_info
import os

os.environ['DISPLAY'] = ':99'
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
os.environ["LIBGL_DRIVERS_PATH"]="/data/liaopan/megprep/softwares/mesa-18.3.6/lib"
os.environ["LD_LIBRARY_PATH"]="/data/liaopan/megprep/softwares/mesa-18.3.6/lib"

#os.environ["LD_PRELOAD"]="/usr/lib64/libstdc++.so.6:/data/liaopan/megprep/softwares/mesa-18.3.6/lib/libGL.so.1"

data_path = mne.datasets.sample.data_path()
# data_path and all paths built from it are pathlib.Path objects
subjects_dir = data_path / "subjects"
subject = "sample"

fname_raw = data_path / "MEG" / subject / f"{subject}_audvis_raw.fif"
info = read_info(fname_raw)
plot_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

fiducials = "estimated"  # get fiducials from fsaverage
coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
fig.plotter.screenshot('debug_icp.png')
