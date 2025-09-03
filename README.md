![MEGPrep Logo](https://github.com/LiaoPan/megprep/blob/main/docs/source/_static/logo.png)
------------------------------------------------

MEGPrep is a robust and fully automated preprocessing pipeline optimized for large-scale MEG studies and compliant with Brain Imaging Data Structure (BIDS) standards.

[![Documentation Status](https://readthedocs.org/projects/megprep/badge/?version=latest)](https://megprep.readthedocs.io/en/latest/?badge=latest)

[![Documentation Status][rtd-badge]][rtd-link]

[rtd-badge]: https://readthedocs.org/projects/megprep/badge/?version=latest
[rtd-link]:  https://megprep.readthedocs.io/en/latest/?badge=latest

[![docs](https://img.shields.io/readthedocs/megprep/stable?logo=readthedocs&label=docs)](https://megprep.readthedocs.io/en/stable/)

## Installation
For installation and usage instructions for MEGPrep, please see the documentation page: https://megprep.readthedocs.io/en/latest/installation.html


## Get Started

### Run MEGPrep via Docker to perform MRI data (T1w) preprocessing
```bash
# Key parameter settings in `nextflow_for_anat.config:`
#       do_fs = true 
#       do_only_anatomy = true
#       anatomy_preprocess_method = "freesurfer" # or "deepprep"

docker run -it --rm \
-v "$(pwd)/<your_bids_dataset>":/input \
-v "$(pwd)/<your_bids_dataset>/derivatives":/output \
-v "$(pwd)/<your_bids_dataset>/smri":/smri \
-v /<your_freesurfer>/license.txt:/fs_license.txt \
-v "$(pwd)/nextflow_for_anat.config":/program/nextflow/nextflow.config \
megprep:<version> \
-i /input \
-o /output \
--fs_license_file /license.txt \
--fs_subjects_dir /smri
```



### Run MEGPrep via Docker to perform MEG data preprocessing.
```bash
docker run -it --rm \
-v "$(pwd)/<your_bids_dataset>":/input \
-v "$(pwd)/<your_bids_dataset>/derivatives":/output \
-v "$(pwd)/<your_bids_dataset>/smri":/smri \
-v /<your_freesurfer>/license.txt:/fs_license.txt \
-v "$(pwd)/nextflow_for_cog.config":/program/nextflow/nextflow.config \
megprep:<version> \
-i /input \
-o /output \
--fs_license_file /license.txt \
--fs_subjects_dir /smri
```

### Launch the MEGPrep interactive report.
```bash
docker run -it -d --rm \
-p 8501:8501 \
-v "$(pwd)/cog_dataset/derivatives":/output \
-v "$(pwd)/cog_dataset/smri":/smri \
megprep:<version> -r
```


## Citation


## License
MEGPre has a MIT license, as found in the [LICENSE](LICENSE) file.


## Bug reports
Please use the [GitHub issue tracker](https://github.com/LiaoPan/megprep/issues) to report bugs.