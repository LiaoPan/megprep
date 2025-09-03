Local
========================

Run MEGPrep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run -it --rm \
        -v /data/datasets/SMN4Lang:/input \
        -v /data/datasets/SMN4Lang/preprocessed:/output \
        -v /data/datasets/SMN4Lang_smri:/smri \
        -v /data/megprep/license.txt:/fs_license.txt \
        -v /data/nextflow.config:/program/nextflow/nextflow.config \
        megprep:0.0.3 \
        -i /input \
        -o /output \
        --fs_license_file /license.txt \
        --fs_subjects_dir /smri \
        --resume

In this command:  


+ ``-it``
   Run in interactive mode, allowing users to interact within the container.  

+ ``--rm``  
   This option automatically removes the container after it exits, ensuring no residual containers remain.  

+ ``-v /data/datasets/SMN4Lang:/input``  
   This option creates a volume mount, mapping the host directory `/data/datasets/SMN4Lang` to the container's `/input` directory, allowing the container to access input data.  

+ ``-v /data/datasets/SMN4Lang/preprocessed:/output``  
   This maps the output directory in the host to the container's `/output` directory for saving processed data.  

+ ``-v /data/datasets/SMN4Lang_smri:/smri``  
   This mounts a directory containing SMRI data(T1w, Freesurfer's SUBJECTS_DIR) to the container's `/smri` directory for application use.

+ ``-v /data/megprep/license.txt:/fs_license.txt``  
   This mounts the FreeSurfer license file into the container, ensuring it has access to the necessary permissions.  

+ ``-v /data/nextflow.config:/program/nextflow/nextflow.config``
   This mounts the Nextflow configuration file so the program inside the container can use it.  

+ ``megprep:0.0.3``  
    This specifies the Docker image and version to run, where `megprep` is the image name, and `0.0.3` is the version.  

+ ``-i /input``  
    This is a parameter passed to the program, specifying the input data directory as `/input`.  

+ ``-o /output``  
    This parameter specifies the output data directory as `/output`.  

+ ``--fs_license_file /fs_license.txt``  
    This passes the path to the FreeSurfer license file to the program, ensuring it can be recognized correctly.  

+ ``--fs_subjects_dir /smri``  
    This specifies the separate SMRI data directory for use by the program.  

+ ``--resume``
    This flag allows the process to resume execution from the last completed step, which is useful for long-running tasks to avoid re-running completed steps.  
