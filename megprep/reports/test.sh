# MEG
export DATASET_REPORT_PATH=/data/liaopan/megprep_demo/cog_dataset/derivatives
export SUBJECTS_DIR=/data/liaopan/megprep_demo/cog_dataset/smri
#OPM
#export DATASET_REPORT_PATH=/data/liaopan/datasets/OPM-COG.v1/derivatives
#export SUBJECTS_DIR=/data/liaopan/datasets/OPM-COG.v1/smri

#Holmes_cn

#auditory_OPM_stationary
#export DATASET_REPORT_PATH=
#export SUBJECTS_DIR=/data/liaopan/datasets/auditory_OPM_stationary/smri

streamlit run reports.py --server.address=0.0.0.0 --server.port=8502 --server.headless=true
