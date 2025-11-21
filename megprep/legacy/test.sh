python /data/liaopan/megprep/megprep/meg_preproc_osl.py \
    --file /data/liaopan/datasets/SQUID-Artifacts/S01.LP.fif \
    --preproc_dir . \
    --config "
    preproc:

    - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}

"

