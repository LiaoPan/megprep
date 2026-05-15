#!/bin/bash  
# Usage:
#$ bash run_for_docker.sh -i /data/liaopan/datasets/Holmes_cn_single/raw --fs_license_file /data/liaopan/megprep/license.txt --fs_subjects_dir /data/liaopan/datasets/Holmes_cn/smri
#  bash run_for_docker.sh -i /data/liaopan/datasets/Holmes_cn_single/raw --fs_license_file /data/liaopan/megprep/license.txt --fs_subjects_dir /data/liaopan/datasets/Holmes_cn/smri -o /data/liaopan/datasets/Holmes_cn_single
# Exit on error
set -e

# Default configuration file and parameters
CONFIG_FILE="/program/nextflow/nextflow.config"
RUN_CONFIG_FILE="/program/nextflow/run_nextflow.config"
INPUT_DIR=""
OUTPUT_DIR=""
STEPS=""
FS_LICENSE_FILE=""
FS_SUBJECTS_DIR=""
T1_DIR=""
T1_INPUT_TYPE=""
ANAT_ONLY=false
MEG_ONLY=false
VIEW_REPORT=false
COHORT_MODE=false
NEXTFLOW_FILE="/program/nextflow/meg_pipeline.nf"
STREAMLIT_APP_PATH="/program/megprep/reports/reports.py"
COHORT_REPORT_PATH="/program/megprep/reports/cohort_static_html_report.py"
STATIC_TASK_LOG_MODE=""
nextflow_args=()

echo "Executor:"
whoami

# Process input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_FILE="$2"; shift ;;
        -i|--input) INPUT_DIR="$2"; shift ;;
        -o|--output) OUTPUT_DIR="$2"; shift ;;
        -s|--steps) STEPS="$2"; shift ;;
        --fs_license_file) FS_LICENSE_FILE="$2"; shift ;;
        --fs_subjects_dir) FS_SUBJECTS_DIR="$2"; shift ;;

        # Other parameters
        --t1_dir) T1_DIR="$2"; shift ;;
        --t1_input_type) T1_INPUT_TYPE="$2"; shift ;;

        # options for specifying only one part
        --anat_only) ANAT_ONLY=true ;;
        --meg_only) MEG_ONLY=true ;;

        # online reports
        -r|--view_report|--view-report) VIEW_REPORT=true ;;

        # cohort mode
        --cohort) COHORT_MODE=true ;;

        # static report options
        --static_task_log_mode|--task-log-mode) STATIC_TASK_LOG_MODE="$2"; shift ;;

        # nextflow options
        --resume) nextflow_args+=("-resume") ;;

        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --config          Specify the Nextflow config file (default: nextflow.config)"
            echo "  -i, --input           Specify the input directory"
            echo "  -o, --output          Specify the output directory(including report results.)"
            echo "  -s, --steps           Same as Nextflow --steps / params.steps (e.g. all, meg_all, anatomy, report, meg_epochs,skip_ica)"
            echo "  -r, --view-report     Run Streamlit to view the report (does not run Nextflow)"
            echo "  --cohort              Treat --input as a directory of datasets; isolate each child's output and FreeSurfer SUBJECTS_DIR"
            echo "  --static_task_log_mode failed|all-command-log|none"
            echo "  --fs_license_file     Specify the FreeSurfer license file"
            echo "  --fs_subjects_dir     Specify the FreeSurfer SUBJECTS_DIR directory containing processed T1 results"
            echo "  --t1_dir              Specify the T1 image directory"
            echo "  --t1_input_type       Specify the T1 input type"
            echo "  --anat_only           Deprecated shortcut for --steps anatomy"
            echo "  --meg_only            Deprecated shortcut for --steps meg_all"
            echo "  --resume              Resume the previous run(nextflow options)"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done


# If --view-report is set, run the Streamlit app instead of Nextflow
if [ "$VIEW_REPORT" = true ]; then
    echo "Starting Streamlit to view the report..."
#    if [ -z "$OUTPUT_DIR" ]; then
#      echo "Output directory must be specified."
#      exit 1
#    fi
#    # set reports path env.
#    export DATASET_REPORT_PATH=$OUTPUT_DIR
    streamlit run "$STREAMLIT_APP_PATH" --server.port=8501 --server.headless=true
    exit 0
fi

# Check if input and output directories are specified
if [ -z "$INPUT_DIR" ]; then
    echo "Input directory must be specified."
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Output directory must be specified."
    exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

read_static_task_log_mode() {
    sed -n 's/^[[:space:]]*static_task_log_mode[[:space:]]*=[[:space:]]*["'\'']\([^"'\'']*\)["'\''].*/\1/p' "$CONFIG_FILE" | head -n 1
}

STATIC_TASK_LOG_MODE="${STATIC_TASK_LOG_MODE:-$(read_static_task_log_mode)}"
STATIC_TASK_LOG_MODE="${STATIC_TASK_LOG_MODE:-failed}"
case "$STATIC_TASK_LOG_MODE" in
    failed|all-command-log|none) ;;
    *)
        echo "Error: invalid --static_task_log_mode '$STATIC_TASK_LOG_MODE' (expected failed, all-command-log, or none)"
        exit 1
        ;;
esac

echo "Using configuration file: $CONFIG_FILE"
echo "Static report task log mode: $STATIC_TASK_LOG_MODE"

write_run_config() {
    local run_input_dir="$1"
    local run_output_dir="$2"
    local run_config_file="$3"
    local run_fs_subjects_dir="${4:-$FS_SUBJECTS_DIR}"
    local run_t1_dir="${5:-$T1_DIR}"

    cp "$CONFIG_FILE" "$run_config_file"

    if [ -n "$run_input_dir" ]; then
        echo "Setting dataset_dir in config to: $run_input_dir"
        sed -i "s|^\s*dataset_dir\s*=.*|    dataset_dir = \"$run_input_dir\"|" "$run_config_file"
    fi

    if [ -n "$run_output_dir" ]; then
        echo "Setting output_dir in config to: $run_output_dir"
        sed -i "s|^\s*output_dir\s*=.*|    output_dir = \"$run_output_dir\"|" "$run_config_file"
    fi

    if [ -n "$run_fs_subjects_dir" ]; then
        echo "Using FreeSurfer subjects directory: $run_fs_subjects_dir"
        mkdir -p "$run_fs_subjects_dir"
        sed -i "s|^\s*fs_subjects_dir\s*=.*|    fs_subjects_dir = \"$run_fs_subjects_dir\"|" "$run_config_file"
    fi

    if [ -n "$FS_LICENSE_FILE" ]; then
        echo "Using FreeSurfer license file: $FS_LICENSE_FILE"
        sed -i "s|^\s*fs_license\s*=.*|    fs_license = \"$FS_LICENSE_FILE\"|" "$run_config_file"
    fi

    if [ -n "$run_t1_dir" ]; then
        echo "Setting t1_dir in config to: $run_t1_dir"
        sed -i "s|^\s*t1_dir\s*=.*|    t1_dir = \"$run_t1_dir\"|" "$run_config_file"
        sed -i "s|^\s*t1_bids_dir\s*=.*|    t1_bids_dir = \"$run_t1_dir\"|" "$run_config_file"
    fi

    if [ -n "$T1_INPUT_TYPE" ]; then
        echo "Setting t1_input_type in config to: $T1_INPUT_TYPE"
        sed -i "s|^\s*t1_input_type\s*=.*|    t1_input_type = \"$T1_INPUT_TYPE\"|" "$run_config_file"
    fi

    echo "Setting static_task_log_mode in config to: $STATIC_TASK_LOG_MODE"
    sed -i "s|^\s*static_task_log_mode\s*=.*|    static_task_log_mode = \"$STATIC_TASK_LOG_MODE\"|" "$run_config_file"
}


# Call Nextflow to run the pipeline with specified configurations
echo "Running Nextflow pipeline..."

steps_args=()
if [ "$ANAT_ONLY" = true ] && [ "$MEG_ONLY" = true ]; then
    echo "Error: --anat_only and --meg_only cannot be used together. Prefer --steps anatomy or --steps meg_all."
    exit 1
fi

if [ -z "$STEPS" ] && [ "$ANAT_ONLY" = true ]; then
    STEPS="anatomy"
    echo "Warning: --anat_only is deprecated; using --steps anatomy."
fi

if [ -z "$STEPS" ] && [ "$MEG_ONLY" = true ]; then
    STEPS="meg_all"
    echo "Warning: --meg_only is deprecated; using --steps meg_all."
fi

if [ -n "$STEPS" ]; then
    echo "Setting steps (Nextflow params.steps): $STEPS"
    steps_args=(--steps "$STEPS")
fi

run_nextflow_pipeline() {
    local run_config_file="$1"
    local run_output_dir="$2"
    local run_work_dir="$3"
    local work_args=()

    mkdir -p "$run_output_dir"
    if [ -n "$run_work_dir" ]; then
        mkdir -p "$run_work_dir"
        work_args=(-w "$run_work_dir")
    fi

    nextflow run "${NEXTFLOW_FILE}" \
        -c "${run_config_file}" \
        "${steps_args[@]}" \
        --static_task_log_mode "$STATIC_TASK_LOG_MODE" \
        "${work_args[@]}" \
        -with-report "${run_output_dir}/report.html" \
        -with-timeline "${run_output_dir}/timeline.html" \
        -with-trace "${run_output_dir}/trace.txt" \
        "${nextflow_args[@]}"

    cp "$run_config_file" "${run_output_dir}/nextflow.config"
}

sanitize_dataset_name() {
    local raw_name="$1"
    raw_name="${raw_name// /_}"
    printf "%s" "$raw_name" | tr -c "A-Za-z0-9_.-" "_"
}

resolve_cohort_t1_dir() {
    local dataset_dir="$1"
    local dataset_name="$2"

    if [ -z "$T1_DIR" ]; then
        printf "%s" "$dataset_dir"
        return
    fi

    if [ -d "${T1_DIR}/${dataset_name}" ]; then
        printf "%s" "${T1_DIR}/${dataset_name}"
        return
    fi

    printf "%s" "$T1_DIR"
}

steps_need_t1() {
    local steps_value="$1"
    local primary="${steps_value%%,*}"

    case "$primary" in
        all|anatomy)
            return 0
            ;;
    esac

    case ",${steps_value}," in
        *,with_anatomy,*)
            return 0
            ;;
    esac

    return 1
}

effective_steps_value() {
    if [ -n "$STEPS" ]; then
        printf "%s" "$STEPS"
        return
    fi
    sed -n 's/^[[:space:]]*steps[[:space:]]*=[[:space:]]*["'\'']\([^"'\'']*\)["'\''].*/\1/p' "$CONFIG_FILE" | head -n 1
}

regenerate_static_report() {
    local run_output_dir="$1"
    if [ -d "${run_output_dir}/preprocessed" ]; then
        python /program/megprep/reports/static_html_report.py \
            --report_root "$run_output_dir" \
            --output_dir "${run_output_dir}/static_html_report" \
            --task_log_mode "$STATIC_TASK_LOG_MODE"
    else
        echo "Skipping static report regeneration; no preprocessed directory found at ${run_output_dir}/preprocessed"
    fi
}

# activate Anaconda virtualenv and virtual display
#/usr/bin/supervisord  -c /etc/supervisor/conf.d/supervisord.conf
#Xvfb :99 -screen 0 1920x1080x24 &
#export DISPLAY=:99
#xhost +
#export QT_QPA_PLATFORM=xcb #offscreen

mkdir -p "$OUTPUT_DIR"

if [ "$COHORT_MODE" = true ]; then
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Error: --cohort requires --input to be a directory containing dataset subdirectories."
        exit 1
    fi

    echo "Running cohort mode. Dataset collection root: $INPUT_DIR"
    datasets_output_dir="${OUTPUT_DIR}/datasets"
    cohort_work_dir="${OUTPUT_DIR}/work"
    cohort_fs_subjects_base="${FS_SUBJECTS_DIR:-/smri}"
    mkdir -p "$datasets_output_dir" "$cohort_work_dir"
    found_dataset=0
    effective_steps="$(effective_steps_value)"

    for dataset_dir in "$INPUT_DIR"/*; do
        if [ ! -d "$dataset_dir" ]; then
            continue
        fi

        original_dataset_name="$(basename "$dataset_dir")"
        dataset_name="$(sanitize_dataset_name "$original_dataset_name")"
        dataset_output_dir="${datasets_output_dir}/${dataset_name}"
        dataset_run_config="${OUTPUT_DIR}/run_nextflow_${dataset_name}.config"
        dataset_fs_subjects_dir="${cohort_fs_subjects_base}/${dataset_name}"
        dataset_t1_dir=""
        if steps_need_t1 "$effective_steps"; then
            dataset_t1_dir="$(resolve_cohort_t1_dir "$dataset_dir" "$original_dataset_name")"
        fi

        echo "Cohort dataset: $dataset_name"
        if [ -n "$dataset_t1_dir" ]; then
            echo "T1: $dataset_t1_dir"
        else
            echo "T1: <not used for steps=${effective_steps:-config default}>"
        fi
        write_run_config "$dataset_dir" "$dataset_output_dir" "$dataset_run_config" "$dataset_fs_subjects_dir" "$dataset_t1_dir"
        run_nextflow_pipeline "$dataset_run_config" "$dataset_output_dir" "${cohort_work_dir}/${dataset_name}"
        regenerate_static_report "$dataset_output_dir"
        found_dataset=1
    done

    if [ "$found_dataset" -eq 0 ]; then
        echo "Error: no dataset subdirectories were found under $INPUT_DIR."
        exit 1
    fi

    python "$COHORT_REPORT_PATH" \
        --cohort_root "$datasets_output_dir" \
        --output_dir "${OUTPUT_DIR}/cohort_static_html_report"

    chmod -R 777 "$OUTPUT_DIR"
    exit 0
fi

write_run_config "$INPUT_DIR" "$OUTPUT_DIR" "$RUN_CONFIG_FILE"
run_nextflow_pipeline "$RUN_CONFIG_FILE" "$OUTPUT_DIR" ""
regenerate_static_report "$OUTPUT_DIR"
chmod -R 777 /output
