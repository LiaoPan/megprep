Outputs
========================
MEGPrep generates several output files and directories, including preprocessing results, quality control reports, and any intermediate files created during the pipeline execution.
The output directory will have a predefined structure that includes:

*   ``preprocessed/``: Contains preprocessed MEG data files.
*   ``work/``: Nextflow working directory.
*   ``artifact_report/``: Contains artifact detection results.
*   ``covariance/``: Contains covariance-related files.
*   ``forward_solution/``: Contains forward model files.
*   ``ica_report/``: Contains ICA results.
*   ``preproc_report/``: Contains preprocessing reports.
*   ``source_recon/``: Contains source reconstruction data.
*   ``trans/``: Contains coregistration-related files.


This structure ensures that users can easily find and navigate to their results.