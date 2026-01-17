#!/bin/bash  
# Usage:
#$ bash run_for_docker.sh -i /data/liaopan/datasets/Holmes_cn_single/raw --fs_license_file /data/liaopan/megprep/license.txt --fs_subjects_dir /data/liaopan/datasets/Holmes_cn/smri
#  bash run_for_docker.sh -i /data/liaopan/datasets/Holmes_cn_single/raw --fs_license_file /data/liaopan/megprep/license.txt --fs_subjects_dir /data/liaopan/datasets/Holmes_cn/smri -o /data/liaopan/datasets/Holmes_cn_single
# Exit on error
# set -e
set +e

# Default configuration file and parameters
CONFIG_FILE="/program/nextflow/nextflow.config"
RUN_CONFIG_FILE="/program/nextflow/run_nextflow.config"
INPUT_DIR=""
OUTPUT_DIR=""
STEPS=""
FS_LICENSE_FILE=""
NEXTFLOW_FILE="/program/nextflow/meg_pipeline.nf"
STREAMLIT_APP_PATH="/program/megprep/reports/reports.py"
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
        --anat_only) ANAT_ONLY=true; shift ;;
        --meg_only) MEG_ONLY=true; shift ;;

        # online reports
        -r|--view_report) VIEW_REPORT=true ;;

        # nextflow options
        --resume) nextflow_args+=("-resume") ;;

        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --config          Specify the Nextflow config file (default: nextflow.config)"
            echo "  -i, --input           Specify the input directory"
            echo "  -o, --output          Specify the output directory(including report results.)"
            echo "  -s, --steps           Specify the steps to execute (e.g., preproc,epoch,source)"
            echo "  -r, --view-report     Run Streamlit to view the report (does not run Nextflow)"
            echo "  --fs_license_file     Specify the FreeSurfer license file"
            echo "  --fs_subjects_dir     Specify the FreeSurfer SUBJECTS_DIR directory containing processed T1 results"
            echo "  --t1_dir              Specify the T1 image directory"
            echo "  --t1_input_type       Specify the T1 input type"
            echo "  --anat_only           Run only the FreeSurfer related steps"
            echo "  --meg_only            Run only the MEG related steps"
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

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi


echo "Using configuration file: $CONFIG_FILE"

cp $CONFIG_FILE $RUN_CONFIG_FILE

# Update dataset_dir and output_dir in the configuration file
if [ ! -z "$INPUT_DIR" ]; then
    echo "Setting dataset_dir in config to: $INPUT_DIR"
    sed -i "s|^\s*dataset_dir\s*=.*|    dataset_dir = \"$INPUT_DIR\"|" "$RUN_CONFIG_FILE"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
    echo "Setting output_dir in config to: $OUTPUT_DIR"
    sed -i "s|^\s*output_dir\s*=.*|    output_dir = \"$OUTPUT_DIR\"|" "$RUN_CONFIG_FILE"
fi

if [ ! -z "$FS_SUBJECTS_DIR" ]; then
    echo "Using FreeSurfer subjects directory: $FS_SUBJECTS_DIR"
    sed -i "s|^\s*fs_subjects_dir\s*=.*|    fs_subjects_dir = \"$FS_SUBJECTS_DIR\"|" "$RUN_CONFIG_FILE"
fi


# Call Nextflow to run the pipeline with specified configurations
echo "Running Nextflow pipeline..."

# activate Anaconda virtualenv and virtual display
#/usr/bin/supervisord  -c /etc/supervisor/conf.d/supervisord.conf
#Xvfb :99 -screen 0 1920x1080x24 &
#export DISPLAY=:99
#xhost +
#export QT_QPA_PLATFORM=xcb #offscreen

nextflow run "${NEXTFLOW_FILE}" \
    -c "${RUN_CONFIG_FILE}" \
    -with-report "${OUTPUT_DIR}/report.html" \
    -with-timeline "${OUTPUT_DIR}/timeline.html" \
    -with-trace \
    "${nextflow_args[@]}"

cp $RUN_CONFIG_FILE "${OUTPUT_DIR}"/nextflow.config
chmod -R 777 /output