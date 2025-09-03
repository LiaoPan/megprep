# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import math
import numpy as np
import mne
from scipy.stats import kurtosis
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, read_ica
from pathlib import Path
import joblib as jl
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl
mne.viz.set_browser_backend('matplotlib')
mpl.rcParams['figure.max_open_warning'] = 100

def add_text(fig_path,exp_var):
    image1 = Image.open(fig_path)  
    
    # Define the text and its formatting
    text = 'mag:{:.1f}'.format(exp_var['mag']*100)+'%'
    # text += ' grad:{:.1f}'.format(exp_var['grad']*100)+'%'
    # font = ImageFont.load_default()  # You can specify a different font if needed    
    font_size = 24  # Set the desired font size
    # font = ImageFont.truetype("/usr/share/fonts/open-sans/OpenSans-Regular.ttf", font_size)
    font = ImageFont.load_default()
    
    # Split the text into words
    words = text.split()
    
    # Add text to the left-top corner of the first image with custom font colors
    draw1 = ImageDraw.Draw(image1)
    
    # Iterate through words and specify custom colors for specific words
    x, y = 10, 2  # Initial position
    for word in words:
        try:
            float_number = abs(float(word))
            if float_number>0.1 and float_number<1:
                text_color = (255, 0, 0)
            # elif float_number>k_thres:
            #     text_color = (255, 0, 0)
            else:
                text_color = (0, 0, 0)
        except ValueError:
            text_color = (0, 0, 0)    
            
        draw1.text((x, y), word, fill=text_color, font=font) 
        x += font.getlength(word) + 5  # Move x position for the next word    
   
    image1.save(fig_path)    
    image1.close()


def run_ica(subj_tag, subj_res_path, subj_res_path_ica, fn_data, fn_ica, n_IC, modality,random_seed):
    figs = []
    subj_res_path_ica = Path(subj_res_path_ica)  
    subj_res_path_ica.mkdir(parents=True, exist_ok=True)  

    raw = mne.io.read_raw_fif(fn_data,preload=True,verbose=False)
    raw_ = raw.copy()

    if not os.path.exists(os.path.join(subj_res_path, fn_ica)):
        ica = ICA(n_components=n_IC, 
                method='fastica',
                max_iter='auto', 
                random_state=random_seed)

        if modality == 'eeg':
            picks = mne.pick_types(raw_.info, meg=False, eeg=True, eog=False,
                                   stim=False, exclude='bads')
            # reject = dict(eeg=200e-6)
        elif modality == 'meg':
            picks = mne.pick_types(raw_.info, meg=True, eeg=False, eog=False,
                                   stim=False, exclude='bads', ref_meg=False)
        elif modality == 'meeg':
            picks = mne.pick_types(raw_.info, meg=True, eeg=True, eog=False,
                                   stim=False, exclude='bads')
        else:
            picks = None

        ica.fit(raw_, picks=picks, reject_by_annotation=True)  # , reject=reject)
        ica.save(os.path.join(subj_res_path, fn_ica))
    else:
        ica = read_ica(os.path.join(subj_res_path, fn_ica))
    
    try:
        labels = mne.read_annotations(fn_data)
        raw_.set_annotations(labels)
    except Exception as e:
        print(e)
    try:
        fig_list = ica.plot_properties(raw_, picks=list(range(ica.n_components_)), reject_by_annotation=True, reject=None,
                                        show=False, verbose=False)
    except Exception as e:
        print(e)

    # explain variance
    explained_var_list = []
    # picks = mne.pick_types(raw_.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False, exclude='bads')
    for t in range(ica.n_components_):
        # explained_var_ratio = ica.get_explained_variance_ratio(raw,components=[t],ch_type=['mag','grad'])
        explained_var_ratio = ica.get_explained_variance_ratio(raw,components=[t],ch_type=['mag'])
        for channel_type, ratio in explained_var_ratio.items():
            print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

        try:
            # tmpPics = subj_res_path_ica / f'_{t}.png'
            tmpPics = subj_res_path_ica / f'{t}_evar_{explained_var_ratio["mag"]:.4f}.png'
            fig_list[t].savefig(tmpPics)
            plt.close(fig_list[t])

            # add explain variance
            add_text(tmpPics, explained_var_ratio)
        except Exception as e:
            print(e)

        explained_var_list.append(explained_var_ratio)

        # add overlay for before and after ICs removal.
        # tmpOverPics = subj_res_path_ica / f"_{t}_overlay.png"
        # fig = ica.plot_overlay(raw, exclude=[t], picks=picks,start=None,stop=20.,title=f"ICA{t:03d}-Signals before (red) and after (black) cleaning") # 20 seconds
        # fig.savefig(tmpOverPics)
        # plt.close(fig)

    # combine png
    # for t in range(ica.n_components_):
    #     img1 = Image.open(f'{subj_res_path_ica}/_{t}.png')
    #     img2 = Image.open(f'{subj_res_path_ica}/_{t}_overlay.png')
    #
    #     total_width = img1.width + img2.width
    #     max_height = max(img1.height, img2.height)
    #
    #     new_img = Image.new('RGBA', (total_width, max_height))
    #
    #     x_offset = 0
    #     for img in [img1, img2]:
    #         new_img.paste(img, (x_offset, 0))
    #         x_offset += img.width
    #
    #     # save and remove verbose png.
    #     new_img.save(f'{subj_res_path_ica}/{t}_evar_{explained_var_list[t]["mag"]:.4f}.png')
    #     os.remove(f'{subj_res_path_ica}/_{t}.png')
    #     os.remove(f'{subj_res_path_ica}/_{t}_overlay.png')
    
    #save explained var list
    fname_ica_explained_var = os.path.join(subj_res_path, 'ica_explained_var.jl')
    jl.dump(explained_var_list, fname_ica_explained_var)

    #save sources 
    fname_ica_sources = os.path.join(subj_res_path, "ica_sources.fif")
    ica_sources = ica.get_sources(raw)

    if 'ctf_head_t' in raw.info:
        info = mne.create_info(ch_names=ica_sources.ch_names, ch_types=ica_sources.get_channel_types(),
                               sfreq=raw.info['sfreq'])
        ica_sources = mne.io.RawArray(ica_sources.get_data(), info)

    ica_sources.save(fname_ica_sources,overwrite=True)


    N = len(ica.unmixing_matrix_)
    dn = 20
    n = math.ceil(N / dn)
    plot_window = 200


    # plot ica comp tc
    for i in range(n):
        if i < n - 1:
            fig = ica.plot_sources(raw_, show=False,
                                    picks=list(np.linspace(dn * i, dn * (i + 1) - 1, dn).astype(int)), start=0,
                                    stop=plot_window, show_scrollbars=False)
            fig.savefig(os.path.join(subj_res_path_ica,
                                        'ica_comp_' + str(dn * i) + '-' + str(dn * (i + 1) - 1) + '_tc.png'), dpi=120)
            plt.close('all')
        else:
            fig = ica.plot_sources(raw_, show=False, picks=list(np.linspace(dn * i, N - 1, N - dn * i).astype(int)),
                                    start=0, stop=plot_window, show_scrollbars=False)
            fig.savefig(os.path.join(subj_res_path_ica, 'ica_comp_' + str(dn * i) + '-' + str(N - 1) + '_tc.png'),
                        dpi=120)
            plt.close('all')

    # plot ica comp topo
    try:
        figs = ica.plot_components(show=False)
        for i in list(np.linspace(0, n - 1, n).astype(int)):
            if i < n - 1:
                figs[i].savefig(os.path.join(subj_res_path_ica,
                                                'ica_comp_' + str(dn * i) + '-' + str(dn * (i + 1) - 1) + '_topo.png'),
                                dpi=120)
            else:
                figs[i].savefig(
                    os.path.join(subj_res_path_ica, 'ica_comp_' + str(dn * i) + '-' + str(N - 1) + '_topo.png'),
                    dpi=120)
        plt.close('all')
    except Exception as e:
        print(e)

    return figs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate ICA components and plots for MEG data.")
    # parser.add_argument('--subject_id', required=True, help='Subject ID')
    # parser.add_argument('--session_id', required=True, help='Session ID')

    parser.add_argument('--raw_file', required=True, help='Path to the raw MEG file')
    parser.add_argument('--num_IC', required=True, type=float, help='The number of ICA components to generate')
    parser.add_argument('--output_dir', required=True, help='Path to save ICA plots and related files')
    parser.add_argument('--seed', required=False, default=2025, help='Random seed for ICA')

    return parser.parse_args()

def main():
    args = parse_arguments()

    subj_tag = f"{Path(args.raw_file).stem}"
    subj_res_path = os.path.join(args.output_dir, f"{Path(args.raw_file).parent.stem}")
    subj_res_path_ica = os.path.join(subj_res_path, "ica_results")
    os.makedirs(subj_res_path_ica, exist_ok=True)

    # subj_path = f"{Path(args.raw_file).parent}"
    subj_ica_file = f"{subj_tag}_ica.fif"
    if not os.path.exists(subj_res_path):
        os.mkdir(subj_res_path)

    num_IC = args.num_IC
    if num_IC.is_integer():
        num_IC = int(num_IC)

    try:
        random_seed = args.seed
        random_seed = int(random_seed)
    except Exception:
        random_seed = 2025

    run_ica(
        subj_tag=subj_tag,
        subj_res_path=subj_res_path,
        subj_res_path_ica=subj_res_path_ica,
        fn_data=args.raw_file,
        fn_ica=subj_ica_file,
        n_IC=num_IC,
        modality="meg",
        random_seed=random_seed,
    )


if __name__ == "__main__":
    main()
