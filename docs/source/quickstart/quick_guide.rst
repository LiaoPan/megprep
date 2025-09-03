Guide for Beginner
=============

Pipeline Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

In the nextflow.config file, you can configure various pipeline parameters to customize your processing workflow:

**nextflow.config**

.. code-block:: groovy

    params {
    dataset_dir = "/input" // Input data directory
    output_dir = "/output" // Output logs directory
    preproc_dir = "${params.output_dir}/preprocessed" // Output results directory
    code_dir = "/program/megprep" // All code for preprocessing
    do_fs = true // Enable FreeSurfer method
    is_bids = true // Whether the data is in BIDS format
    anatomy_preprocess_method = "freesurfer" // "freesurfer" or "deepprep"
    anatomy_select_tag = "_run-02_T1w" // Anatomy selection tag

       // FreeSurfer parameters
       t1_dir = "/input"                 // T1 image directory
       t1_input_type = "nifti"           // Input type: 'dicom' or 'nifti'
       fs_subjects_dir = "/smri"         // FreeSurfer subjects directory
       freesurfer_home = "/opt/freesurfer/7.4.1-1" // FreeSurfer installation path

       // DeepPrep parameters
       deepprep_device = "cpu"           // Device for DeepPrep processing
       t1_bids_dir = "/input"            // BIDS directory for T1 images
       fs_license = "/data/megprep/megprep/tools/DeepPrep/license.txt" // License for DeepPrep

       // MEG import datasets parameters
       dataset_format = 'auto'           // Dataset format: 'auto', 'bids', or 'raw'
       file_suffix = '.fif'              // File suffix for raw datasets

       // BEM model parameters
       bem_config = """
           ico: 4
           conductivity:
               - 0.3
       """

       // MEG artifacts detection parameters
       artifact_config = """
           - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
           - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
       """

       meg_type = "grad"                 // MEG type: "grad" or "mag"
       segment_len = 1000                // Segment length for detection

       // Preprocessing parameters[osl-ephys]
       preproc_config = """
           preproc:
           - filter: {l_freq: 0.5, h_freq: 40, method: iir, iir_params: {order: 5, ftype: butter}}
           - notch_filter: {freqs: 50 100}
           - resample: {sfreq: 100}
       """

       // ICA parameters
       num_IC = 60                       // Number of independent components
       ICA_output_dir = "ica_report"     // Report location for ICA results

       // Epochs parameters
       epoch_output_dir = "epochs"       // Output directory for epochs
       epoch_config = """
       task_type: 'task'                 // Type of task: 'task' or 'resting'

       resting:
           fixed_length_duration: 2.0     // Duration for resting fixed length

       event_source: 'find_events'        // Source of events for epochs

       find_events:
           stim_channel: null
           shortest_event: 1
           min_duration: 0.0
       epochs:
           event_id: null                 // Event ID for epochs
           tmin: -0.2                     // Start time for each epoch
           tmax: 1                        // End time for each epoch
           reject_by_annotation: true
           picks: meg
           baseline: null
           reject:
               grad: 4000e-13
               mag: 4e-12
           preload: true
           detrend: null
       """

       // Coregistration parameters
       trans_output_dir = "trans"
       core_config = """
       omit_head_shape_points: 1 # mm
       grow_hair: 0.0 # mm
       icp:
           n_iterations: 200
           lpa_weight: 1.0
           nasion_weight: 10.0
           rpa_weight: 1.0
           hsp_weight: 10.0
           eeg_weight: 0.0
           hpi_weight: 1.0
       finetune_icp:
           n_iterations: 200
           lpa_weight: 0.0
           nasion_weight: 0.0
           rpa_weight: 0.0
           hsp_weight: 10.0
           eeg_weight: 0.0
           hpi_weight: 0.0
       """

       // Covariance parameters
       covar_output_dir = "covariance"
       covar_config = """
           events:
               stim_channel: null
               shortest_event: 1
               min_duration: 0.0
           epochs:
               event_id: null
               tmin: -0.2
               tmax: 0.0
               reject_by_annotation: true
               picks: meg
               baseline: null
               reject:
                   grad: 4000e-13
                   mag: 4e-12
               preload: true
               detrend: null
               reject_by_annotation: true
           covariance:
               tmin: null
               tmax: null
               rank: null  // Rank used for covariance calculation
       """

       // Forward solution parameters
       fwd_output_dir = "forward_solution"
       fwd_epoch_label = "wdonset"
       fwd_config = """
           surface: white
           spacing: ico4
       """

       // Source imaging parameters
       src_output_dir = "source_recon"
       src_config = """
           source_methods:
               - dSPM

           data_type: meg
           spacing: ico4
           epoch_label: wdonset

           dSPM:
               inverse_operator:
                   loose: auto
                   depth: 0.8
                   fixed: auto
                   rank:
                       meg: 50
               apply_inverse:
                   method: dSPM
                   pick_ori: normal

           LCMV:
               n_rank: 50
               cov_tmin: 0.01
               cov_tmax: 0.4
               make_lcmv:
                   reg: 0.05
                   pick_ori: null
                   rank:
                       meg: 50
                   weight_norm: unit-noise-gain-invariant
       """

Run MEGPrep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run -it --rm \
        -v /data/datasets/SMN4Lang:/input \
        -v /data/datasets/SMN4Lang/test_v3.5:/output \
        -v /data/datasets/SMN4Lang_smri:/smri \
        -v /data/megprep/license.txt:/fs_license.txt \
        -v /data/megprep/nextflow/nextflow.config:/program/nextflow/nextflow.config \
        megprep:0.0.3 \
        -i /input \
        -o /output \
        --fs_license_file /license.txt \
        --fs_subjects_dir /smri \
        --resume
