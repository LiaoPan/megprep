![MEGPrep Logo](https://github.com/LiaoPan/megprep/blob/main/docs/source/_static/logo.png)
------------------------------------------------

MEGPrep is a robust and fully automated preprocessing pipeline optimized for large-scale MEG studies and compliant with Brain Imaging Data Structure (BIDS) standards.

[![Documentation Status](https://readthedocs.org/projects/megprep/badge/?version=latest)](https://megprep.readthedocs.io/en/latest/?badge=latest)

## Installation
For installation and usage instructions for MEGPrep, please see the documentation page: https://megprep.readthedocs.io/en/latest/quickstart/installation.html




## Get Started


### MegPrep Docker Image Pull Guide

#### Prerequisites
-------------

- Docker installed on your system (Linux, Windows, or macOS). For installation instructions, refer to the official Docker documentation: https://docs.docker.com/engine/install/
- A computer with internet access.
- Administrative privileges (if needed for Docker operations).
- At least 8GB of RAM recommended for running Docker containers.


#### Pulling the MegPrep Image
-------------------------

With Docker installed, you can pull the ``megprep:<version>`` image from Docker Hub.

1. Open a terminal (or Command Prompt/PowerShell on Windows).

2. Run the pull command:

```bash
$ docker pull megprep:<version>
```

3. Verify the image is downloaded:

```bash
$ docker images
```

   You should see ``megprep`` with tag ``<version>`` in the list.


#### Running the Container
---------------------

To run the container after pulling:

```bash
$ docker run megprep:<version> -h
```

For more help, visit https://docs.docker.com/ or the [MEGPrep documentation](megprep.readthedocs.io/en/latest/).

### MEGPrep Parameters Settings

For the MEGPrep parameter settings, please refer to [MEGPrep Parameters Template](nextflow/nextflow.config).

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