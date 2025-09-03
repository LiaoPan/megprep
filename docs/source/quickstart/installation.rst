Installation
=============

To install MEGPrep, you need to ensure that Docker are installed on your system. Please follow these steps for installation:


Install Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install Docker according to your operating system. For detailed installation instructions, please visit the `Docker official website <https://docs.docker.com/get-docker/>`_.

.. code-block:: bash
    $ docker info # Confirm that docker is installed correctly


.. `Install Singularity <https://docs.sylabs.io/guides/3.5/user-guide/index.html>`_ [options]


Download MEGPrep Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker pull megprep:<version>


See the parameter description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

docker run megprep:<version> -h
Usage: /program/nextflow/run.sh [options]
Options:
  -c, --config          Specify the Nextflow config file (default: nextflow.config)
  -i, --input           Specify the input directory
  -o, --output          Specify the output directory(including report results.)
  -r, --view-report     Run Streamlit to view the report (does not run Nextflow)
  --fs_license_file     Specify the FreeSurfer license file
  --fs_subjects_dir     Specify the FreeSurfer SUBJECTS_DIR directory containing processed T1 results
  --t1_dir              Specify the T1 image directory
  --t1_input_type       Specify the T1 input type
  --anat_only           Run only the FreeSurfer related steps
  --meg_only            Run only the MEG related steps
  --resume              Resume the previous run(nextflow options)