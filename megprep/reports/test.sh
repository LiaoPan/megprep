#export DATASET_REPORT_PATH=/data/liaopan/megprep_demo/cog_dataset/derivatives
#export SUBJECTS_DIR=/data/liaopan/megprep_demo/cog_dataset/smri
export DATASET_REPORT_PATH=/data/liaopan/datasets/OPM-COG.v1/derivatives
export SUBJECTS_DIR=/data/liaopan/datasets/OPM-COG.v1/smri


streamlit run reports.py --server.address=0.0.0.0 --server.port=8502 --server.headless=true
