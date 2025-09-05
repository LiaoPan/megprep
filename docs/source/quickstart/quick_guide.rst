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
    do_only_anatomy = false // if true, only anatomy preprocessing.
    is_bids = true // Whether the data is in BIDS format
    meg_visualize = true // for coregistration and source_recon

    // MRI Parameters
    anatomy_preprocess_method = "freesurfer" // "freesurfer" or "deepprep"
    anatomy_select_tag = "" //"_run-02_T1w" // defalut:"", <subject_id>+<anatomy_tag> to select multiple runs T1.

    // MRI Import datasets
    mri_import_config = """
        # Filter out specific anatomy, only bids support.
        subject_id: '001' # str or null
        session_id: null # str or null
        task: null
        run_id: null
    """

    //freesurfer method
    t1_dir = "/input"               // T1 image directory
    t1_input_type = "nifti"         // 'dicom' or 'nifti'

    //deepprep method
    deepprep_device = "cpu"
    t1_bids_dir = "/input"
    fs_license = "/data/liaopan/megprep/megprep/tools/DeepPrep/license.txt"

    //[fixed params]
    fs_subjects_dir = "/smri" // FREESURFER SUBJECTS_DIR in Container[fixed]
    freesurfer_home = "/opt/freesurfer/7.4.1-1"


    // MEG import datasets parameters
    dataset_format = 'auto'           // Dataset format: 'auto', 'bids', or 'raw'
    file_suffix = '.fif'              // File suffix for raw datasets
    meg_import_config = """
        # Filter out specific megs, only bids support.
        subject_id: null # str or null
        session_id: null # str or null
        task: null
        run_id: null
        #subject_id:
        #    - '001'
        #session_id:
        #    - '006'
        #task:
        #    - compr
    """

    // BEM model parameters
    bem_config = """
        ico: 4
        conductivity:
            - 0.3
    """
    // MEG Preprocessing Parameters
    osl_random_seed = 2025
    preproc_config = """
        preproc:
        #- set_channel_types:  {EEG061: eog, EEG062: eog, EEG063: ecg}
        - filter: {l_freq: 0.5, h_freq: 40, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 100}
        #- bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        #- bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
    """


    // MEG Artifacts Detection Parameters
    artifact_config = """
        find_bad_channels:
            pyprep:
                deviation:
                    deviation_threshold: 5.0
                snr: {}
                nan_flat: {}
                # hfnoise:
                #     HF_zscore_threshold: 5.0
              # ransac: # very slow
                #     n_samples: 50
                #     sample_prop: 0.25
                #     corr_thresh: 0.75
                #     frac_bad: 0.4
                #     corr_window_secs: 5.0
                #     channel_wise: true
                #     max_chunk_size: null
                # correlation:
                #     correlation_secs: 1.0
                #     correlation_threshold: 0.4
                #     frac_bad: 0.01
            psd:
                std_multiplier: 6
            osl:
                ref_meg: auto
                significance_level: 0.05
            mne:
                find_bad_channels_lof:
                    n_neighbors: 20
                    picks: mag
                    metric: euclidean
                    threshold: 1.5

        find_bad_segments:
            osl:
                segment_len: 1000 # detect_badsegments
            mne:
                annotate_muscle_zscore:
                    ch_type: mag
                    threshold: 12
                #annotate_amplitude:
                #    picks: meg
                # annotate_break:
                #     min_break_duration: 15.0
                #     t_start_after_previous: 5.0
                #     t_stop_before_next: 5.0
    """


    // MEG ICA Parameters
    num_IC = 60 // 0.99999
    ICA_random_seed = 2025
    ICA_output_dir = "ica_report" // relative path based on preproc dir

    // MEG ICA Label Parameters
    ic_label_config = """
        # detect artifact ICs
        ic_ecg: true
        ic_eog: true
        ic_outlier: true # detect artifact ICs by rules.

        find_bads_eog:
            ch_name: null # or the ch_name of EOG.
            threshold: auto
            l_freq: 1
            h_freq: 10
            start: null
            stop: null
            measure: zscore

        find_bads_ecg:
            ch_name: null # or the ch_name of ECG.
            threshold: auto
            method: ctps
            l_freq: 8
            h_freq: 16
            measure: zscore

        find_bads_muscle:
            threshold: 0.5
            start: null
            stop: null
            l_freq: 7
            h_freq: 45

        ICA_classify:
            meg_vendor: ctf # neuromag or ctf or quanmag_opm or quspin_opm
            explained_var:
                threshold: 0.1
                ch_type: mag
            find_ecg_ics:
                time_segment: 10 # seconds
                ts_ecg_num_max: 20 # Maximum number of heartbeats expected in the chosen time segment
                l_freq: 0.1
                h_freq: 10
                peak_threshod_coef: 0.4 #Indicates the threshold of the number of ecg signal peak interval (unit: index). (peak_threshod = 0.4 * fs) | # for 1 seconds
                peak_std_threshold_coef: 0.05 #Standard deviation threshold of ecg signal peak interval (unit: index). (peak_std_threshold = peak_std_threshold_coef * fs) | # for 1 seconds
            find_abnormal_psd_ics:
                attention_low_freq: 0
                attention_high_freq: 150
                le_low_freq: 0
                le_high_freq: 12
                low_freq_energy_threshold: 0.8 # Threshold above which the component is flagged by low-frequency energy ratio
    """

    // MEG Epochs Parameters
    epoch_output_dir = "epochs" // relative path based on preproc dir
    epoch_config = """
    task_type: 'task'   # or 'resting'

    resting:
        fixed_length_duration: 2.0

    event_source: 'find_events'  # 'event_file' or 'find_events'

    autoreject: false  # true or false| automatic global_rejection_threshold, get the `reject` params.

    #event_file：specific the event type of *_events.tsv | filter | the value of `null` means to get all events.
    event_file:
        trial_type: null
        #type: # you can change `trial_type` to `type` or other type related.
        #    word_onset_01: 1
        #    phoneme_onset_01: 2
        # trial_type:
        #    - word_onset_01
        #    - phoneme_onset_01

    # find events
    find_events:
        #stim_channel: UPPT001 # for CTF Holmes
        stim_channel: null
        shortest_event: 1
        min_duration: 0.0
    epochs:
        event_id: null
        tmin: -0.2
        tmax: 1
        reject_by_annotation: true
        picks: meg
        baseline: null
        #reject:
            #grad: 4000e-13
            #mag: 4e-12
        preload: true
        detrend: null
    """


    // MEG-MRI coregistraion Parameters
    trans_output_dir = "trans"
    core_config = """
    omit_head_shape_points: 1 # mm
    grow_hair: 0.0 #mm
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

    // Covariance Parameters
    covar_output_dir = "covariance"
    covar_visualize = true // Whether to generate covariance graphs
    covar_type = "epochs" // raw or epochs
    raw_covariance_task_id = "resting" // task name
    covar_config = """
        ## 1.Estimate noise covariance matrix from a continuous segment of raw data.
        compute_raw_covariance:
            tmin: 0
            tmax: null
            method: auto
            reject:
                grad: 4000e-13  # T / m (gradiometers)
                mag: 4e-12  # T (magnetometers)
            reject_by_annotation: true
            rank: info

        ## 2.Estimate noise covariance matrix from epochs.
        # find events
        events:
            stim_channel: null
            #stim_channel: UPPT001 # for CTF Holmes
            shortest_event: 1
            min_duration: 0.0

        # For baseline epochs
        epochs:
            event_id: null # baseline event id
            tmin: -0.2 # Start time (in seconds) for covariance calculation window
            tmax: 0.0 # End time (in seconds) for covariance calculation window
            picks: meg
            baseline: null
            #reject:
                #grad: 4000e-13
                #mag: 4e-12
            preload: true
            detrend: null
            reject_by_annotation: true

        covariance:
            tmin: null  #Start time for baseline. If null start at first sample.
            tmax: null  # End time for baseline. If null end at last sample.
            rank: null  # Rank used for covariance calculation| meg: 90
    """


    // Forward Solution Parameters
    fwd_output_dir = "forward_solution"
    fwd_epoch_label = "wdonset"
    fwd_config = """
        surface: white # pial
        spacing: ico4
    """

    // Source Imaging Parameters
    src_output_dir = "source_recon"
    src_type = "epochs" // raw or epochs
    src_config = """
        source_methods:
            - dSPM

        data_type: meg  # mag
        spacing: ico4
        epoch_label: wdonset

        dSPM:
            inverse_operator:
                loose: auto
                depth: 0.8
                fixed: auto
                rank: info
                    #meg : 50
            apply_inverse:
                lambda2: 0.1111111111
                method: dSPM
                pick_ori: normal

        LCMV:
            n_rank: 50  # compute_covariance,meg's n_rank
            cov_tmin: 0.01
            cov_tmax: 0.4
            make_lcmv:
                reg: 0.05
                pick_ori: null
                rank:
                    meg : 50
                weight_norm: unit-noise-gain-invariant
"""

Run MEGPrep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run -it --rm \
        -v /data/datasets/SMN4Lang:/input \
        -v /data/datasets/SMN4Lang/preprocessed:/output \
        -v /data/datasets/SMN4Lang/smri:/smri \
        -v /data/megprep/license.txt:/fs_license.txt \
        -v /data/megprep/nextflow/nextflow.config:/program/nextflow/nextflow.config \
        megprep:0.0.3 \
        -i /input \
        -o /output \
        --fs_license_file /license.txt \
        --fs_subjects_dir /smri \
        --resume
