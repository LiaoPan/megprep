#!/usr/bin/env nextflow
// Run with a project-specific config, for example:
// nextflow run nextflow/meg_anat_pipeline_for_docker.nf -c nextflow/nextflow.config
nextflow.enable.dsl=2

import groovy.json.JsonOutput
import groovy.json.JsonSlurper
// include { deepprep } from '/opt/DeepPrep/deepprep/nextflow/deepprep.nf'

log.info "MEGPrep Anatomy and MEG Preprocessing Pipeline"
log.info "============================="
log.info ""
log.info "Start time: $workflow.start"
log.info ""

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}

// =============================================================================
// MRI anatomy preprocessing
// =============================================================================


process import_MRI_dataset {
    input:
    path dataset_dir

    output:
    path 'imported_t1_data.txt', emit: imported_t1_data

    script:
    script_name = "${params.code_dir}/mri_import_dataset.py"
    """
    mkdir -p ${params.preproc_dir}
    python ${script_name} \\
        --bids_dir ${dataset_dir} \\
        --config "${params.mri_import_config}" \\
        --output_file imported_t1_data.txt
    """
}

process read_nifti_files {

    input:
    path input_file

    output:
    tuple val(sub_id), path(nii_file),emit: output_ch

    script:
    """
    while IFS= read -r line; do
        sub_id=\$(echo "\$line" | awk -F : '{print \$1}' | xargs)
        nii_files=\$(echo "\$line" | awk -F : '{print \$2}' | sed "s/[][]//g" | xargs)

        IFS=',' read -ra files <<< "\$nii_files"
        for nii_file in "\${files[@]}"; do
            echo "\$sub_id \$nii_file" | xargs
        done
    done < ${input_file}
    """
}

// Convert T1-weighted DICOM data to NIfTI.
process dcm2niix {
    tag "DICOM to Nifti"

    input:
    path t1_dicom_dir

    output:
    path "${t1_dicom_dir}/${t1_dicom_dir}.nii.gz"

    script:
    dcm_tag = "1004-t1_mprage_sag_iso_mww64CH_S3_DIS3D"
    output_file = "${t1_dicom_dir}/${t1_dicom_dir}.nii.gz"

    """
    echo "Starting DICOM to NIfTI conversion for directory: ${t1_dicom_dir}"

    # Check if the output file already exists, if it does, skip the conversion
    if [ -f ${output_file} ]; then
        echo "Output file ${output_file} already exists. Skipping conversion."
    else
        echo "Output file does not exist. Starting conversion."
        dcm2niix -o ${t1_dicom_dir} -z y -f ${t1_dicom_dir} ${t1_dicom_dir}/${dcm_tag}
    fi

    echo "Finished DICOM to NIfTI conversion."
    """
}


// Run FreeSurfer recon-all and build the head surface.
process run_freesurfer {
    tag "${subject_name}"

    input:
    val anat_file
    val subject_name
    path fs_subjects_dir

    output:
    path "${fs_subjects_dir}/${subject_name}", emit: fs_subject_dir
    path "${fs_subjects_dir}/${subject_name}/mri/seghead.mgz"
    path "${fs_subjects_dir}/${subject_name}/surf/lh.seghead"
    val(subject_name),emit: mri_subject_id

    script:
    """
    if [ ! -d "${params.fs_subjects_dir}" ]; then
        echo "Directory ${params.fs_subjects_dir} does not exist. Creating it now."
        mkdir -p "${params.fs_subjects_dir}"
    fi

    recon-all -sd ${params.fs_subjects_dir} -all -i $anat_file -s $subject_name
    recon-all -sd ${params.fs_subjects_dir} -all -s $subject_name -3T -openmp 4
    mkheadsurf -sd ${params.fs_subjects_dir} -s $subject_name -srcvol T1.mgz -thresh1 30
    """
}

// Run DeepPrep as an alternative anatomical preprocessing backend.
process run_deepprep {
    tag "${subject_name}"
//     debug true
//     publishDir "${fs_subjects_dir}", mode: 'copy'

    input:
    path anat_bids_dir
    val subject_name
    path fs_subjects_dir
    path preproc_dir
//     val mri_subject_name

    output:
//     path "${preproc_dir}/deepprep/Recon/*"
    val(subject_name),emit:mri_subject_id
    path "${fs_subjects_dir}/*/mri/brain.mgz"
    path "${fs_subjects_dir}/*/surf/lh.pial"


    script:

//     /opt/DeepPrep/deepprep/deepprep.sh  ${params.t1_bids_dir} ${output_dir} participant \
//         --participant_label ${subject_name} \
//         --skip_bids_validation \
//         --anat_only \
//         --fs_license_file /fs_license.txt \
//         --device ${params.deepprep_device} \
//         --subject_id ${params.mri_select_subject_id} \
//         --session_id ${params.mri_select_session_id} \
//         --task ${params.mri_select_task} \
//         --run_id ${params.mri_select_run_id} \
//         --resume

    output_dir="${params.preproc_dir}/deepprep"
    """
    if [ ! -d "${params.fs_subjects_dir}" ]; then
        echo "Directory ${params.fs_subjects_dir} does not exist. Creating it now."
        mkdir -p "${params.fs_subjects_dir}"
    fi

    /opt/DeepPrep/deepprep/deepprep.sh  ${params.t1_bids_dir} ${output_dir} participant \
        --participant_label ${subject_name} \
        --skip_bids_validation \
        --anat_only \
        --fs_license_file /fs_license.txt \
        --device ${params.deepprep_device} \
        --mri_import_config "${params.mri_import_config}" \
        --resume

    kill -9 \$(pgrep redis-server)
    cp -rf ${output_dir}/Recon/* ${fs_subjects_dir}/
    """
}

// Build the FreeSurfer head surface after external reconstruction.
process run_mkheadsurf {
    tag "${subject_name}"

    input:
    val subject_name
    path fs_subjects_dir

    output:
    path "${fs_subjects_dir}/${subject_name}", emit: fs_subject_dir
    path "${fs_subjects_dir}/${subject_name}/mri/seghead.mgz"
    path "${fs_subjects_dir}/${subject_name}/surf/lh.seghead"
    val(subject_name),emit: mri_subject_id

    script:
    """
    mkheadsurf -sd ${fs_subjects_dir} -s $subject_name -srcvol T1.mgz -thresh1 30
    """
}

// Generate the BEM model from the anatomical reconstruction.
process generate_bem {
    tag "${subject_name}"

    input:
    val subject_dir
    val subject_name
    val config
    path fs_subjects_dir

    output:
    path "${fs_subjects_dir}/${subject_basename}/bem/*"
    path "${fs_subjects_dir}", emit: fs_subjects_dir
    tuple val(subject_name),path("${fs_subjects_dir}"), emit: mri_subject_id

    script:
    script_name = "${params.code_dir}/generate_bem.py"
    subject_basename = file(subject_dir).getBaseName()
    """
    python3 ${script_name} \\
        --subject_dir ${subject_dir} \\
        --config "${params.bem_config}" \\
        --output_dir ${fs_subjects_dir}/${subject_basename}/bem
    """
}


// =============================================================================
// MEG preprocessing
// =============================================================================

// Discover MEG input files from the configured dataset.
process import_MEG_dataset {

    input:
    path dataset_dir
    val dataset_format
    val file_suffix

    output:
    path 'imported_meg_data.txt',emit: imported_meg_data

//     publishDir "${params.preproc_dir}", mode: 'copy'

    script:
    script_name = "${params.code_dir}/meg_import_dataset.py"
    """
    mkdir -p ${params.preproc_dir}
    python ${script_name} \\
        --dataset_dir ${dataset_dir} \\
        --dataset_format ${dataset_format} \\
        --file_suffix ${file_suffix} \\
        --output_file imported_meg_data.txt \\
        --config "${params.meg_import_config}"
    """
}

process meg_preproc_osl {
    tag "${raw_subject_basename}"

    memory { 6.GB * task.attempt }
    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 3

    input:
    path dataset_dir
    val raw_subject_path
    path preproc_dir
    val preproc_config

    output:
    path "${preproc_dir}/${raw_subject_basename}/${raw_subject_basename}_preproc-raw${params.file_suffix}"

    script:
    script_name = "${params.code_dir}/meg_preproc_osl.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    """
    python ${script_name} \\
        --file ${raw_subject_path} \\
        --preproc_dir ${preproc_dir} \\
        --seed ${params.osl_random_seed} \\
        --config "${preproc_config}"
    """
}


// Detect bad channels and bad time segments.
process detect_Artifacts {
    tag "${raw_subject_basename}"

    input:
    path preproc_dir
    val raw_subject_path

    output:
    path "${preproc_dir}/artifact_report/${raw_subject_parent}/*_bad_channels.txt",emit: bad_channels
    path "${preproc_dir}/artifact_report/${raw_subject_parent}/*_bad_segments.txt",emit: bad_segments // MNE annotations
    val "${raw_subject_path}",emit: preproc_subject_paths

    script:
    script_name = "${params.code_dir}/meg_detect_artifacts.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_parent = file(raw_subject_path).getParent().getBaseName()
    """
    mkdir -p ${preproc_dir}/artifact_report/${raw_subject_parent}
    echo ${raw_subject_path}
    python ${script_name} \\
        --input ${raw_subject_path} \\
        --output ${preproc_dir}/artifact_report/${raw_subject_parent} \\
        --config "${params.artifact_config}"
    """
}



process run_ICA {
    tag "${raw_subject_basename}"

    memory { 8.GB * task.attempt }
    errorStrategy { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries 3

    input:
    path preproc_dir
    val raw_subject_path
    path bad_channels
    path bad_segments

    output:
    path "${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/ica_results/*.png"
    path "${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/ica_explained_var.jl",emit:ica_expvars
    path "${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/ica_sources.fif",emit: ica_sources
    path "${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/*_ica.fif",emit: ica_fif_paths
    val "${raw_subject_path}",emit: preproc_subject_paths

    script:
    script_name = "${params.code_dir}/run_ica.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()
    """
    mkdir -p ica_report
    python ${script_name} \\
        --raw_file $raw_subject_path \\
        --output_dir ${preproc_dir}/${params.ICA_output_dir} \\
        --num_IC ${params.num_IC} \\
        --fname_bad_channels ${bad_channels} \\
        --fname_bad_segments ${bad_segments} \\
        --seed ${params.ICA_random_seed}
    """
}

process run_IC_label {
//     debug true
    tag "${raw_subject_basename}"

    input:
    path preproc_dir
    val raw_subject_path
    path ica_file_path
    path ica_source
    path ica_expvar

    output:
    path "${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/marked_components.txt",emit:marked_components
    val "${raw_subject_path}",emit: preproc_subject_paths

    script:
    script_name = "${params.code_dir}/run_ica_label.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()
    """
    python ${script_name} \\
        --raw_data_path ${raw_subject_path} \\
        --ica_file ${ica_file_path} \\
        --output_dir ${preproc_dir}/${params.ICA_output_dir} \\
        --config "${params.ic_label_config}"
    """
}

process apply_ICA {
    tag "${raw_subject_basename}"

    input:
    path preproc_dir
    val raw_subject_path
    path marked_components
    path bad_channels
    path bad_segments

    output:
    path "${preproc_dir}/${raw_subject_dir_basename}/${raw_subject_basename}_clean_raw.fif",emit: preproc_subject_paths
    tuple val("${target_mri_subject_id}"),val("${preproc_dir}/${raw_subject_dir_basename}/${raw_subject_basename}_clean_raw.fif"), emit: target_mri_subject_id

    script:
    script_name = "${params.code_dir}/apply_ica.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()
    target_mri_subject_id = raw_subject_basename.split('_')[0] + params.anatomy_select_tag
    """
    python ${script_name} \\
        --raw_file ${raw_subject_path} \\
        --ica_file ${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/${raw_subject_basename}_ica.fif \\
        --exclude_file ${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename}/marked_components.txt \\
        --output_file ${preproc_dir}/${raw_subject_dir_basename}/${raw_subject_basename}_clean_raw.fif \\
        --output_dir ${preproc_dir}/${params.ICA_output_dir}/${raw_subject_dir_basename} \\
        --fname_bad_channels ${bad_channels} \\
        --fname_bad_segments ${bad_segments}
    """
}


process epochs {
    tag "${raw_subject_basename}"

    input:
    path dataset_dir
    path preproc_dir
    val raw_subject_path
    val events_files

    output:
    val "${preproc_dir}/${params.epoch_output_dir}/${raw_subject_dir_basename}/${raw_subject_basename}-epo.fif",emit: preproc_subject_paths
    tuple val("${raw_subject_dir_basename}"),val("${preproc_dir}/${params.epoch_output_dir}/${raw_subject_dir_basename}/${raw_subject_basename}-epo.fif"),emit: meg_subject_id

    script:
    script_name = "${params.code_dir}/epochs.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    filtered_raw_subject_basename = file(raw_subject_path).getBaseName().replace("_meg_preproc-raw_clean_raw", "").replace("_meg_preproc-raw", "")
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()

    dataset_dir_parent_dir = file(dataset_dir).getParent()

    events_file = events_files.find { event_file ->
            event_file.contains(filtered_raw_subject_basename)
        }

    """
    mkdir -p ${preproc_dir}/${params.epoch_output_dir}/${raw_subject_dir_basename}
    python ${script_name} \\
        --preproc_raw_file ${raw_subject_path} \\
        --events_file ${events_file} \\
        --output_epoch_file ${raw_subject_basename}-epo.fif \\
        --output_dir ${preproc_dir}/${params.epoch_output_dir}/${raw_subject_dir_basename} \\
        --config "${params.epoch_config}"
    """
}


process coregistration {
    tag "${raw_subject_dir_basename}"

    time '1h'
    errorStrategy { task.exitStatus in 137..140 || task.exitStatus == 1 ? 'retry' : 'terminate' }
    maxRetries 6

    input:
    path preproc_dir
    tuple val(mri_subject_id),path(fs_subjects_dir),val(raw_subject_path)

    output:
//     val "${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename}/coreg-trans.fif",emit: trans_files
    path "${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename}/coreg-trans.fif",emit: trans_files
    tuple val("${raw_subject_dir_basename}"),val("${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename}/coreg-trans.fif"),emit: meg_subject_id

    script:
    script_name = "${params.code_dir}/coregistration.py"
    println "raw_subject_path: ${raw_subject_path}"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()
    // Make sure the MRI has the same name as MEG
    if (mri_subject_id == null) {
        mri_subject_id = raw_subject_basename.split('_')[0]
        }

//     println("[coregistration]Extracted Subject ID: ${mri_subject_id}")
    """
    python ${script_name} \\
        --raw_file ${raw_subject_path} \\
        --subjects_dir ${fs_subjects_dir}/${mri_subject_id} \\
        --visualize ${params.meg_visualize} \\
        --output_dir ${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename} \\
        --config "${params.core_config}"
    """
}


process coregistrations {
    // condition: do_fs=false
    tag "${raw_subject_dir_basename}"

    time '1h'
    errorStrategy { task.exitStatus in 137..140  || task.exitStatus == 1 ? 'retry' : 'terminate' }
    maxRetries 6

    input:
    path preproc_dir
    val raw_subject_path
    path fs_subjects_dir

    output:
    path "${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename}/coreg-trans.fif",emit: trans_files
    tuple val("${raw_subject_dir_basename}"),val("${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename}/coreg-trans.fif"),emit: meg_subject_id

    script:
    script_name = "${params.code_dir}/coregistration.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()
    // Make sure the MRI has the same name as MEG
    mri_subject_id = raw_subject_basename.split('_')[0]
//     println("Extracted Subject ID: ${mri_subject_id}")
    """
    python ${script_name} \\
        --raw_file ${raw_subject_path} \\
        --subjects_dir ${fs_subjects_dir}/${mri_subject_id} \\
        --visualize ${params.meg_visualize} \\
        --output_dir ${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename} \\
        --config "${params.core_config}"
    """
}

// Estimate covariance from epoched data.
process compute_covariances {
    tag "${raw_subject_basename}"

//     time '6h' // timeout
    errorStrategy { task.exitStatus in 137..140 || task.exitStatus == 1 ? 'retry' : 'terminate' }
    maxRetries 3

    input:
    path preproc_dir
    val raw_subject_path

    output:
    path "${preproc_dir}/${params.covar_output_dir}/${raw_subject_dir_basename}/bl-cov.fif",emit: bl_cov_files
    tuple val("${raw_subject_dir_basename}"),val("${preproc_dir}/${params.covar_output_dir}/${raw_subject_dir_basename}/bl-cov.fif"),emit: meg_subject_id

    script:
    script_name = "${params.code_dir}/compute_covariance.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()
    """
    python ${script_name} \\
        --raw_data_file ${raw_subject_path} \\
        --output_dir ${preproc_dir}/${params.covar_output_dir}/${raw_subject_dir_basename} \\
        --visualize ${params.covar_visualize} \\
        --covar_type ${params.covar_type} \\
        --config "${params.covar_config}"
    """
}

// Estimate covariance from continuous raw data.
process compute_covariance {
    tag "${raw_subject_dir_basename}"

//     time '6h' // timeout
    errorStrategy { task.exitStatus in 137..140 || task.exitStatus == 1 ? 'retry' : 'terminate' }
    maxRetries 3

    input:
    path preproc_dir
    tuple val(raw_subject_path),val(raw_data_file) // baseline raw data.

    output:
    path "${preproc_dir}/${params.covar_output_dir}/${raw_subject_dir_basename}/bl-cov.fif",emit: bl_cov_files
    tuple val("${raw_subject_dir_basename}"),val("${preproc_dir}/${params.covar_output_dir}/${raw_subject_dir_basename}/bl-cov.fif"),emit: meg_subject_id


    script:
    script_name = "${params.code_dir}/compute_covariance.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()

//     raw_path_str = raw_subject_path.toString()
//     raw_data_file = (params.covar_type == "raw")
//                         ? raw_path_str.replaceAll(/task-[^_]+/, "task-${params.raw_covariance_task_id}")
//                         : raw_path_str
    """
    python ${script_name} \\
        --raw_data_file ${raw_data_file} \\
        --output_dir ${preproc_dir}/${params.covar_output_dir}/${raw_subject_dir_basename} \\
        --visualize ${params.covar_visualize} \\
        --covar_type ${params.covar_type} \\
        --config "${params.covar_config}"
    """
}

process forward_solution {
    tag "${raw_subject_dir_basename}"

    input:
    path preproc_dir
//     val raw_subject_path // trans reuslts of coregistration
    path fs_subject_dir
    tuple val(meg_subject_id),val(trans_path),val(epoch_path)

    output:
    path "${preproc_dir}/${params.fwd_output_dir}/${raw_subject_dir_basename}/*-fwd.fif",emit: fwd_files
    path "${epoch_file}",emit: epoch_files
    path "${raw_file}",emit: raw_files
    tuple val("${raw_subject_dir_basename}"),val("${epoch_file}"),val("${raw_file}"),emit: meg_subject_id

    script:
    raw_subject_path = trans_path
    script_name = "${params.code_dir}/forward_solution.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()

//     println("raw_subject_path:${raw_subject_path},raw_subject_basename:${raw_subject_basename}")
    mri_subject_id = raw_subject_dir_basename.split('_')[0]
//     if (fs_subjects_dir == null) {
//         fs_subject_dir = "${params.fs_subjects_dir}"
//         }
    mri_subject_dir = fs_subject_dir/mri_subject_id
//     println("Extracted Subject ID: ${mri_subject_id}")


//     trans_file = "${preproc_dir}/${params.trans_output_dir}/${raw_subject_dir_basename}/coreg-trans.fif"
//     epoch_file = "${preproc_dir}/${params.epoch_output_dir}/${raw_subject_dir_basename}/${raw_subject_dir_basename}_preproc-raw_clean_raw-epo.fif"
    trans_file = trans_path
    epoch_file = epoch_path
    raw_file = "${preproc_dir}/${raw_subject_dir_basename}/${raw_subject_dir_basename}_preproc-raw_clean_raw.fif"
    """
    mkdir -p ${preproc_dir}/${params.fwd_output_dir}/${raw_subject_dir_basename}
    python ${script_name} \\
        --epoch_file ${epoch_file} \\
        --epoch_label ${params.fwd_epoch_label}  \\
        --output_dir ${preproc_dir}/${params.fwd_output_dir}/${raw_subject_dir_basename} \\
        --trans_file ${trans_file} \\
        --mri_subject_dir ${mri_subject_dir} \\
        --config "${params.fwd_config}"
    """
}


process source_imaging {
    tag "${raw_subject_dir_basename}"

    errorStrategy { task.exitStatus in 137..140 || task.exitStatus == 1 ? 'retry' : 'ignore' }
    maxRetries 6


    input:
    val src_type
    path preproc_dir
//     val raw_subject_path
//     val bl_cov_file
    tuple val(meg_subject_id),val(epoch_path),val(raw_path),val(bl_cov_file)

    output:
    path "${preproc_dir}/${params.src_output_dir}/${raw_subject_dir_basename}/*.stc",emit: source_files

    script:
    if (src_type == 'epochs') {
        raw_subject_path = epoch_path
    } else if (src_type == 'raw') {
        raw_subject_path = raw_path
    } else {
        error "Invalid src_type: ${src_type}. Please specify 'epochs' or 'raw'."
    }
    script_name = "${params.code_dir}/source_localization.py"
    raw_subject_basename = file(raw_subject_path).getBaseName()
    raw_subject_dir_basename = file(raw_subject_path).getParent().getBaseName()

    """
    mkdir -p ${preproc_dir}/${params.src_output_dir}/${raw_subject_dir_basename}
    python ${script_name} \\
        --data_mode ${params.src_type} \\
        --data_file ${raw_subject_path}  \\
        --fs_subjects_dir ${params.fs_subjects_dir} \\
        --output_dir ${preproc_dir}/${params.src_output_dir}/${raw_subject_dir_basename} \\
        --forward_dir ${preproc_dir}/${params.fwd_output_dir} \\
        --visualize ${params.meg_visualize} \\
        --noise_covariance_dir ${preproc_dir}/${params.covar_output_dir} \\
        --config "${params.src_config}"
    """

}

// Build the dataset-level portable static HTML report.
process generate_static_html_report {
    tag "static-html-report"

    input:
    val source_artifacts
    path run_manifest
    val preserve_existing_manifest

    output:
    path "static_html_report_done.txt", emit: completion_marker

    script:
    report_script = "${params.code_dir}/reports/static_html_report.py"
    report_root = params.preproc_dir
    report_output_dir = "${params.output_dir}/static_html_report"
    zip_output = false
    """
    set -euo pipefail
    manifest_dir="${params.preproc_dir}/logs"
    manifest_dst="\${manifest_dir}/megprep_run_manifest.json"
    manifest_backup=""
    mkdir -p "\${manifest_dir}"

    if [ "${preserve_existing_manifest}" = "true" ] && [ -f "\${manifest_dst}" ]; then
        manifest_backup="\${manifest_dst}.before_report_only.\$\$"
        cp "\${manifest_dst}" "\${manifest_backup}"
    fi

    restore_manifest() {
        if [ "${preserve_existing_manifest}" = "true" ]; then
            if [ -n "\${manifest_backup}" ] && [ -f "\${manifest_backup}" ]; then
                mv "\${manifest_backup}" "\${manifest_dst}"
            else
                rm -f "\${manifest_dst}"
            fi
        fi
    }
    trap restore_manifest EXIT

    # When the workflow wrote manifest directly under preprocessed/logs, Nextflow stages it in
    # the work dir as a symlink to the same path as manifest_dst; cp then errors (same file).
    if [ ! "${run_manifest}" -ef "\${manifest_dst}" ]; then
        cp "${run_manifest}" "\${manifest_dst}"
    fi

    python "${report_script}" \\
        --report_root "${report_root}" \\
        --output_dir "${report_output_dir}" \\
        --bad_channel_threshold ${params.bad_channel_threshold} \\
        --bad_segment_threshold ${params.bad_segment_threshold} \\
        --coreg_mean_threshold ${params.coreg_mean_threshold} \\
        --coreg_max_threshold ${params.coreg_max_threshold} \\
        --epoch_reject_rate_threshold ${params.epoch_reject_rate_threshold} \\
        --zip_output ${zip_output}

    echo "Static HTML report generated at ${report_output_dir}" > static_html_report_done.txt
    """
}

import java.nio.file.Path
class AnatOutput {
    String mri_subject_id
    Path fs_subjects_dir

    // Keep anatomy outputs available for both channel-backed and pre-existing anatomy modes.
    AnatOutput(String mri_subject_id, Path fs_subjects_dir) {
        this.mri_subject_id = mri_subject_id
        this.fs_subjects_dir = fs_subjects_dir
    }
}



/**
 * Parse params.steps into pipeline flags.
 * Primary: report | anatomy | all | meg_all | meg_artifacts | meg_ica | meg_epochs (aliases: meg, artifacts, ica, epochs)
 * Modifiers: skip_ica (meg_epochs only), with_anatomy (meg_* only; not meg_all)
 */
Map parseMegPipelineSteps(String stepsRaw) {
    def parts = stepsRaw.split(',').collect { it.trim().toLowerCase() }.findAll { it.size() > 0 }
    if (!parts) {
        throw new IllegalArgumentException("params.steps is empty")
    }
    def aliases = [meg: 'meg_all', artifacts: 'meg_artifacts', ica: 'meg_ica', epochs: 'meg_epochs']
    def primary = aliases.containsKey(parts[0]) ? aliases[parts[0]] : parts[0]
    def mods = parts.size() > 1 ? parts[1..-1].collect { it.trim().toLowerCase() }.toSet() : [] as Set

    def allowedMods = ['skip_ica', 'with_anatomy'] as Set
    mods.each { m ->
        if (!allowedMods.contains(m)) {
            throw new IllegalArgumentException("Unknown steps modifier: ${m}. Allowed: skip_ica, with_anatomy")
        }
    }

    if (primary == 'meg_all' && mods.contains('with_anatomy')) {
        throw new IllegalArgumentException("steps=meg_all cannot be combined with with_anatomy; use steps=all or meg_*,with_anatomy")
    }

    def skipIca = mods.contains('skip_ica')
    def withAnatomy = mods.contains('with_anatomy')

    int megStage = -1
    boolean runAnatomy = false
    boolean runMeg = false

    switch (primary) {
        case 'report':
            break
        case 'anatomy':
            runAnatomy = true
            break
        case 'all':
            runAnatomy = true
            runMeg = true
            megStage = 3
            break
        case 'meg_all':
            runMeg = true
            megStage = 3
            break
        case 'meg_artifacts':
            runMeg = true
            megStage = 0
            runAnatomy = withAnatomy
            break
        case 'meg_ica':
            runMeg = true
            megStage = 1
            runAnatomy = withAnatomy
            break
        case 'meg_epochs':
            runMeg = true
            megStage = 2
            runAnatomy = withAnatomy
            break
        default:
            throw new IllegalArgumentException("Unknown steps '${primary}'. Use: report, anatomy, all, meg_all, meg_artifacts, meg_ica, meg_epochs (aliases: meg, artifacts, ica, epochs).")
    }

    if (skipIca && megStage != 2) {
        throw new IllegalArgumentException("skip_ica is only supported with meg_epochs (e.g. steps=meg_epochs,skip_ica). Full all/meg_all requires ICA-clean raw for forward/source.")
    }

    return [primary: primary, megStage: megStage, runAnatomy: runAnatomy, runMeg: runMeg, skipIca: skipIca]
}

workflow {
    def cfg = parseMegPipelineSteps(params.steps ?: 'meg_all')

    log.info "Pipeline steps: primary=${cfg.primary}, megStage=${cfg.megStage}, runAnatomy=${cfg.runAnatomy}, runMeg=${cfg.runMeg}, skipIca=${cfg.skipIca}"

    def snapshot = [
            dataset_dir: params.dataset_dir?.toString(),
            output_dir: params.output_dir?.toString(),
            preproc_dir: params.preproc_dir?.toString(),
            code_dir: params.code_dir?.toString(),
            fs_subjects_dir: params.fs_subjects_dir?.toString(),
            anatomy_preprocess_method: params.anatomy_preprocess_method?.toString(),
            dataset_format: params.dataset_format?.toString(),
            covar_type: params.covar_type?.toString(),
            src_type: params.src_type?.toString(),
            is_bids: params.is_bids
    ]
    def partialManifest = [
        manifest_schema_version: 2,
        steps_raw: (params.steps ?: 'meg_all').toString(),
        parsed: [
            primary: cfg.primary,
            meg_stage: cfg.megStage,
            run_anatomy: cfg.runAnatomy,
            run_meg: cfg.runMeg,
            skip_ica: cfg.skipIca
        ],
        params_snapshot: snapshot
    ]
    if (cfg.primary == 'report') {
        partialManifest.report_only = true
        def mf = new File("${params.preproc_dir}/logs/megprep_run_manifest.json")
        if (mf.exists()) {
            try {
                def prev = new JsonSlurper().parse(mf) as Map
                def p = prev?.parsed as Map
                if (p && p.primary?.toString() != 'report' && p.run_meg == true) {
                    partialManifest.parsed = p
                    partialManifest.params_snapshot = (prev.params_snapshot instanceof Map) ? prev.params_snapshot : snapshot
                    partialManifest.pipeline_steps_raw = (prev.pipeline_steps_raw ?: prev.steps_raw)?.toString()
                }
            } catch (Exception e) {
                log.warn "Could not merge prior megprep_run_manifest.json: ${e.message}"
            }
        }
    }
    partialManifest['workflow_meta'] = [
        session_id: workflow.sessionId?.toString() ?: '',
        run_name: workflow.runName?.toString() ?: '',
        start: workflow.start?.toString() ?: '',
        nextflow_version: workflow.nextflow?.version?.toString() ?: '',
        launch_dir: workflow.launchDir?.toString() ?: '',
        project_dir: workflow.projectDir?.toString() ?: ''
    ]
    def manifestJson = JsonOutput.prettyPrint(JsonOutput.toJson(partialManifest))
    // Write manifest synchronously in the workflow driver (no extra process) so the console
    // task list matches older runs and local executor noise stays lower.
    def run_manifest_ch
    if (cfg.primary == 'report') {
        def scratchDir = new File("${params.output_dir}")
        scratchDir.mkdirs()
        def scratchManifest = new File(scratchDir, 'megprep_run_manifest.report_only.json')
        scratchManifest.setText(manifestJson, 'UTF-8')
        run_manifest_ch = Channel.value(file(scratchManifest.absolutePath))
    } else {
        def logsDirForManifest = new File("${params.preproc_dir}/logs")
        logsDirForManifest.mkdirs()
        def publishedManifest = new File(logsDirForManifest, 'megprep_run_manifest.json')
        publishedManifest.setText(manifestJson, 'UTF-8')
        run_manifest_ch = Channel.value(file(publishedManifest.absolutePath))
    }

    // Snapshot Nextflow config(s) into preprocessed/logs (best-effort).
    // Prefer workflow.configFiles when available so custom `-c /path/to/config` and Docker's
    // /program/nextflow/run_nextflow.config are captured before the static report is built.
    // Fallback to launchDir/projectDir conventional names for older Nextflow metadata.
    try {
        def logDir = new File("${params.preproc_dir}/logs")
        if (!logDir.exists() && !logDir.mkdirs()) {
            log.warn "Could not create preprocessed/logs for nextflow.config snapshot: ${logDir.absolutePath}"
        } else {
            def launchDir = new File(workflow.launchDir.toString())
            def projectDir = new File(workflow.projectDir.toString())
            def copiedNames = new HashSet()
            def copiedPairs = new HashSet()

            def copyConfig = { File src, String targetName, boolean overwrite ->
                if (!src || !src.exists() || !src.isFile()) {
                    return false
                }
                def pairKey = "${src.canonicalPath}->${targetName}"
                if (copiedPairs.contains(pairKey)) {
                    copiedNames.add(targetName)
                    return false
                }
                def dst = new File(logDir, targetName)
                if (!overwrite && dst.exists()) {
                    copiedNames.add(targetName)
                    return false
                }
                if (src.canonicalPath != dst.canonicalPath) {
                    dst.bytes = src.bytes
                }
                copiedPairs.add(pairKey)
                copiedNames.add(targetName)
                log.info "nextflow.config snapshot: ${src.absolutePath} -> ${dst.absolutePath}"
                return true
            }

            try {
                def configFiles = workflow.configFiles
                if (configFiles) {
                    def configFileItems = (configFiles instanceof Collection) ? configFiles : (configFiles.getClass().isArray() ? configFiles.toList() : [configFiles])
                    configFileItems.each { cfgPath ->
                        def src = new File(cfgPath.toString())
                        def srcName = src.name
                        if (srcName in ['nextflow.config', 'run_nextflow.config']) {
                            copyConfig(src, srcName, true)
                        } else {
                            copyConfig(src, srcName, true)
                            // static_html_report.py looks for run_nextflow.config / nextflow.config.
                            copyConfig(src, 'run_nextflow.config', true)
                        }
                    }
                }
            } catch (Exception ignored) {
                log.debug "workflow.configFiles is not available in this Nextflow version"
            }

            [launchDir, projectDir].each { baseDir ->
                ['nextflow.config', 'run_nextflow.config'].each { name ->
                    copyConfig(new File(baseDir, name), name, false)
                }
            }

            if (copiedNames.isEmpty()) {
                log.warn(
                    "nextflow.config snapshot: no nextflow.config or run_nextflow.config found under " +
                    "workflow.configFiles, launchDir=${launchDir.absolutePath}, or projectDir=${projectDir.absolutePath}. " +
                    "Static report will fall back to other paths if available."
                )
            }
        }
    } catch (Exception e) {
        log.warn "nextflow.config snapshot skipped: ${e.message}"
    }

    if (cfg.primary == 'report') {
        generate_static_html_report(Channel.value(true), run_manifest_ch, true)
    } else {

    if (cfg.runAnatomy) {
        // Anatomy preprocessing workflow.
        if (params.is_bids) {
            println("params.is_bids:${params.is_bids}")

            t1_files = import_MRI_dataset(params.t1_dir)

            t1_files.imported_t1_data
                .splitText()
                .map { it.trim() }
                .filter { it }
                .map { line ->
                    def matcher = line =~ /sub-(\d+):\[(.+?)\]/
                    if (matcher) {
                        def paths = matcher[0][2].split(',').collect { it.trim().replaceAll(/'/, '') }
                        return paths
                    } else {
                        return []
                    }
                }
                .flatten()
                .set { t1_files_path }

                t1_nifti_files = t1_files_path.filter { it.endsWith(".nii.gz") }

                t1_files = t1_nifti_files

                subject_names = t1_nifti_files.map { filePath ->
                        def matcher = filePath =~ /sub-\d+/
                        matcher ? matcher[0] : ''
                    }
                    subject_names = subject_names.unique()

                subject_names.view { "Subject Names: ${it}" }


            if (params.anatomy_preprocess_method == 'freesurfer') {

                    fs_recon = run_freesurfer(t1_files, subject_names, params.fs_subjects_dir)

                    fs_anatomy_output = generate_bem(fs_recon.fs_subject_dir,fs_recon.mri_subject_id, params.bem_config, params.fs_subjects_dir)

            } else if (params.anatomy_preprocess_method == 'deepprep') {

                    fs_recon = run_deepprep(params.t1_bids_dir, subject_names, params.fs_subjects_dir, params.preproc_dir)

                    subject_names = Channel.fromPath("${params.fs_subjects_dir}/*", type: 'dir')
                                        .filter { !it.name.startsWith("fsaverage") }
                                        .map { it.getBaseName() }
                    subject_names.view { name -> println "MRI Subject name: ${name}" }

                    fs_recon_mk = run_mkheadsurf(fs_recon.mri_subject_id,params.fs_subjects_dir)
                    fs_anatomy_output = generate_bem(fs_recon_mk.fs_subject_dir,fs_recon_mk.mri_subject_id,params.bem_config, params.fs_subjects_dir)
            } else {
                error "Unsupported anatomy preprocessing method: ${params.anatomy_preprocess_method}. Supported methods are 'freesurfer' and 'deepprep'."
            }
        } else {
            if (params.t1_input_type == 'dicom') {
                println "params.t1_dir: ${params.t1_dir}"
                t1_dicom_dirs = Channel.fromPath("${params.t1_dir}/*/", type: 'dir')
                log.info "DICOM to NifTI..."
                dcm2niix(t1_dicom_dirs)
                t1_files = dcm2niix.out
            } else if (params.t1_input_type == 'nifti') {
                t1_nifti_files = Channel.fromPath("${params.t1_dir}/*.{nii,nii.gz}")
                t1_nifti_files.view { "T1 NIfTI Files: $it" }
                t1_files = t1_nifti_files
            } else {
                error "Unsupported t1_input_type: ${params.t1_input_type}. Supported types are 'dicom' and 'nifti'."
            }

            subject_names = t1_files.map { it.name.replace(".nii.gz", "") }
            fs_recon = run_freesurfer(t1_files, subject_names, params.fs_subjects_dir)

            fs_anatomy_output = generate_bem(fs_recon.fs_subject_dir, fs_recon.mri_subject_id, params.bem_config, params.fs_subjects_dir)

        }

    } else {
        fs_subjects_dir = file(params.fs_subjects_dir)
        fs_anatomy_output = new AnatOutput(null,fs_subjects_dir)
    }

    if ( cfg.runMeg ) {
       raw_files = import_MEG_dataset(params.dataset_dir,params.dataset_format,params.file_suffix)

       raw_files.imported_meg_data
               .splitText()
               .map { it.trim() }
               .filter { it }
               .set {raw_files_path}

       preproc_subject_paths = meg_preproc_osl(params.dataset_dir,raw_files_path,params.preproc_dir,params.preproc_config)

      preproc_subject_paths_dta = detect_Artifacts(params.preproc_dir,preproc_subject_paths)

       if (cfg.megStage >= 1 && !cfg.skipIca) {
       preproc_subject_paths_ica = run_ICA(params.preproc_dir,
                                        preproc_subject_paths_dta.preproc_subject_paths,
                                        preproc_subject_paths_dta.bad_channels,
                                        preproc_subject_paths_dta.bad_segments)

       preproc_subject_paths_ica_label = run_IC_label(params.preproc_dir,
                                            preproc_subject_paths_ica.preproc_subject_paths,
                                            preproc_subject_paths_ica.ica_fif_paths,
                                            preproc_subject_paths_ica.ica_sources,
                                            preproc_subject_paths_ica.ica_expvars)

       preproc_subject_paths_clean = apply_ICA(params.preproc_dir,
                                            preproc_subject_paths_ica_label.preproc_subject_paths,
                                            preproc_subject_paths_ica_label.marked_components,
                                            preproc_subject_paths_dta.bad_channels,
                                            preproc_subject_paths_dta.bad_segments)
       }

        events_files = raw_files_path.collect { raw_subject_path ->
            return raw_subject_path.replaceAll(/_meg\..*/, '_events.tsv')
        }

        if (cfg.megStage >= 2) {
        if (cfg.skipIca) {
            if (params.covar_type == 'raw'){
                preproc_subject_paths
                    .filter { orig_path_obj ->
                        def orig_path = orig_path_obj.toString()
                        !orig_path.contains("task-${params.raw_covariance_task_id}")
                    }
                    .set { preproc_subject_raw }
            } else {
                preproc_subject_raw = preproc_subject_paths
            }
        } else {
        if (params.covar_type == 'raw'){
            preproc_subject_paths_clean.preproc_subject_paths
                    .filter { orig_path_obj ->
                        def orig_path = orig_path_obj.toString()
                        !orig_path.contains("task-${params.raw_covariance_task_id}")
                    }
                    .set { preproc_subject_raw }
        } else {
            preproc_subject_raw = preproc_subject_paths_clean.preproc_subject_paths
        }
        }

        epoch_subject_paths = epochs(params.dataset_dir, params.preproc_dir,preproc_subject_raw,events_files)
        }

        if (cfg.megStage >= 3) {

        if (params.covar_type == 'epochs'){
            cov_files = compute_covariances(params.preproc_dir,preproc_subject_paths_clean.preproc_subject_paths)
        } else if (params.covar_type == 'raw'){
            preproc_subject_paths_clean.preproc_subject_paths
                .map { orig_path_obj ->
                    def orig_path = orig_path_obj.toString()

                    if (orig_path.contains("task-${params.raw_covariance_task_id}"))
                        return null

                    def raw_data_file = orig_path.replaceAll(/task-[^_]+/, "task-${params.raw_covariance_task_id}")
                    tuple(orig_path_obj, raw_data_file)
                }
                .filter { it != null }
                .filter { orig_path_obj, raw_data_file ->
                    new File(raw_data_file).exists()
                }
                .set { valid_subject_raws_ch }

            cov_files = compute_covariance(params.preproc_dir, valid_subject_raws_ch)
        } else {
            error "Unsupported covar_type: ${params.covar_type}."
        }

        target_subject_id = preproc_subject_paths_clean.target_mri_subject_id
        if (fs_anatomy_output?.mri_subject_id) {
            println "fs_anatomy_output.mri_subject_id is set."

            core_inputs = fs_anatomy_output.mri_subject_id.combine(target_subject_id, by: 0)
            core_inputs.view {"coregistration: ${it} "}

            trans_subject_paths = coregistration(params.preproc_dir,core_inputs)

            epoch_subject_id = epoch_subject_paths.meg_subject_id
            fwd_inputs = trans_subject_paths.meg_subject_id.combine(epoch_subject_id, by: 0)

            fwds = forward_solution(params.preproc_dir,
                            params.fs_subjects_dir,
                            fwd_inputs)

            cov_subject_id = cov_files.meg_subject_id
            source_inputs = fwds.meg_subject_id.combine(cov_subject_id, by: 0)

            source_results = source_imaging(params.src_type, params.preproc_dir, source_inputs)

            generate_static_html_report(source_results.source_files.collect(), run_manifest_ch, false)

        } else {
                println "fs_anatomy_output.mri_subject_id is not set."
                trans_subject_paths = coregistrations(params.preproc_dir,preproc_subject_raw,fs_anatomy_output.fs_subjects_dir)

                epoch_subject_id = epoch_subject_paths.meg_subject_id
                fwd_inputs = trans_subject_paths.meg_subject_id.combine(epoch_subject_id, by: 0)

                fwds = forward_solution(params.preproc_dir,
                                params.fs_subjects_dir,
                                fwd_inputs)

                cov_subject_id = cov_files.meg_subject_id
                source_inputs = fwds.meg_subject_id.combine(cov_subject_id, by: 0)

                source_results = source_imaging(params.src_type, params.preproc_dir, source_inputs)

                generate_static_html_report(source_results.source_files.collect(), run_manifest_ch, false)
        }
        } else if (cfg.megStage == 2) {
            generate_static_html_report(epoch_subject_paths.meg_subject_id.collect(), run_manifest_ch, false)
        } else if (cfg.megStage == 1) {
            generate_static_html_report(preproc_subject_paths_clean.preproc_subject_paths.collect(), run_manifest_ch, false)
        } else if (cfg.megStage == 0) {
            generate_static_html_report(preproc_subject_paths_dta.bad_channels.collect(), run_manifest_ch, false)
        }
    }
    }
}
