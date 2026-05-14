#!/usr/bin/env bash
set -euo pipefail

# Cohort runner: each immediate child directory under DATASET_ROOT is treated as
# one MEGPrep dataset (Nextflow run + per-dataset reports), then a cohort-level
# static HTML report is built from those outputs.
#
# Paths below are placeholders. Override before running, for example:
#   DATASET_ROOT=/path/to/megqc_cohort INPUT \
#   OUTPUT_ROOT=/path/to/megqc_cohort_OUTPUT \
#   bash run_MultiDatasets.sh

PIPELINE="${PIPELINE:-nextflow/meg_anat_pipeline_for_docker.nf}"
CONFIG="${CONFIG:-nextflow/nextflow_for_cohort.config}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/megqc_cohort_INPUT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/megqc_cohort_OUTPUT}"
FS_SUBJECTS_ROOT="${FS_SUBJECTS_ROOT:-${OUTPUT_ROOT}/smri}"
T1_ROOT="${T1_ROOT:-}"
STEPS="${STEPS:-meg_ica}"
RESUME="${RESUME:--resume}"

sanitize_dataset_name() {
    local raw_name="$1"
    raw_name="${raw_name// /_}"
    printf "%s" "$raw_name" | tr -c "A-Za-z0-9_.-" "_"
}

resolve_t1_dir() {
    local dataset_dir="$1"
    local original_dataset_name="$2"

    if [ -z "$T1_ROOT" ]; then
        printf "%s" "$dataset_dir"
        return
    fi

    if [ -d "${T1_ROOT}/${original_dataset_name}" ]; then
        printf "%s" "${T1_ROOT}/${original_dataset_name}"
        return
    fi

    printf "%s" "$T1_ROOT"
}

if [ ! -d "$DATASET_ROOT" ]; then
    echo "DATASET_ROOT does not exist: $DATASET_ROOT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/datasets" "$OUTPUT_ROOT/work" "$FS_SUBJECTS_ROOT"

found_dataset=0
for dataset_dir in "$DATASET_ROOT"/*; do
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi

    original_dataset_name="$(basename "$dataset_dir")"
    dataset_name="$(sanitize_dataset_name "$original_dataset_name")"
    dataset_output_dir="${OUTPUT_ROOT}/datasets/${dataset_name}"
    dataset_preproc_dir="${dataset_output_dir}/preprocessed"
    dataset_work_dir="${OUTPUT_ROOT}/work/${dataset_name}"
    dataset_fs_subjects_dir="${FS_SUBJECTS_ROOT}/${dataset_name}"
    dataset_t1_dir="$(resolve_t1_dir "$dataset_dir" "$original_dataset_name")"

    mkdir -p "$dataset_output_dir" "$dataset_work_dir" "$dataset_fs_subjects_dir"

    echo "============================================================"
    echo "Dataset: $original_dataset_name"
    echo "Input:   $dataset_dir"
    echo "Output:  $dataset_output_dir"
    echo "MRI:     $dataset_fs_subjects_dir"
    echo "T1:      $dataset_t1_dir"
    echo "============================================================"

    nextflow run "$PIPELINE" \
        -c "$CONFIG" \
        -w "$dataset_work_dir" \
        --steps "$STEPS" \
        --dataset_dir "$dataset_dir" \
        --output_dir "$dataset_output_dir" \
        --preproc_dir "$dataset_preproc_dir" \
        --fs_subjects_dir "$dataset_fs_subjects_dir" \
        --t1_dir "$dataset_t1_dir" \
        --t1_bids_dir "$dataset_t1_dir" \
        -with-report "${dataset_output_dir}/report.html" \
        -with-timeline "${dataset_output_dir}/timeline.html" \
        -with-trace \
        $RESUME

    found_dataset=1
done

if [ "$found_dataset" -eq 0 ]; then
    echo "No dataset subdirectories were found under: $DATASET_ROOT" >&2
    exit 1
fi

python megprep/reports/cohort_static_html_report.py \
    --cohort_root "${OUTPUT_ROOT}/datasets" \
    --output_dir "${OUTPUT_ROOT}/cohort_static_html_report"

echo "Cohort report: ${OUTPUT_ROOT}/cohort_static_html_report/index.html"
