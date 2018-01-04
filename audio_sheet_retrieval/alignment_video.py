#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:59:00 2017

@author: matthias
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.colormaps import cmaps
import seaborn as sns

sns.set_style('ticks')
cmap = sns.color_palette("colorblind")

SPEC_CONTEXT = 42


if __name__ == "__main__":
    """ main """
    
    # load data
    data = pickle.load(open("res_a2s_align/alignment_dump_mozart_ccal_cont2_est_UV_pydtw.pkl", "rb"))
    spec, sheet, a2s_mapping, dtw_res = data
    
#    plt.figure("Sheet")
#    plt.subplots_adjust(left=0.01, right=0.99)
#    plt.imshow(sheet, cmap=plt.cm.gray)
#    
#    plt.figure("Spec")
#    plt.subplots_adjust(left=0.01, right=0.99)
#    plt.imshow(spec)
#    
#    plt.show(block=True)
    
    # iterate frames
    fs = SPEC_CONTEXT // 2
    fe = spec.shape[1] - fs
    for i, frame_id in enumerate(xrange(fs, fe)):
        print frame_id
        pxl_coord = a2s_mapping[frame_id]
        
        context = 500
        x_min = np.max([0, pxl_coord - context])
        x_max = np.min([x_min + 2 * context, sheet.shape[1] - 1])
        x_min = x_max - 2 * context
        
        plt.figure("Alignment", figsize=(10, 10))
        plt.clf()
        
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 2])
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.90, hspace=0.05, wspace=0.05)
        
        plt.subplot(gs[0])
        plt.imshow(sheet, cmap=plt.cm.gray)
        plt.plot(2 * [pxl_coord], [0, sheet.shape[0]], '-', color=cmap[0], linewidth=5, alpha=0.8)
        plt.xlim([x_min, x_max])
        plt.ylim([sheet.shape[0] - 1, 0])
        plt.axis("off")
        plt.title("Sheet Image", fontsize=20)
        
        plt.subplot(gs[1])
        excerpt = spec[:, frame_id - fs : frame_id + fs]
        plt.imshow(excerpt, cmap=cmaps["viridis"], origin='lower')
        plt.plot(2 * [fs], [0, spec.shape[0] - 1], 'w-', linewidth=3.0, alpha=0.8)
        plt.xlim([0, excerpt.shape[1] - 1])
        plt.ylim([0, excerpt.shape[0] - 1])
        plt.axis("off")
        plt.title("Spectrogram", fontsize=20)
        
        plt.subplot(gs[2])
        plt.imshow(dtw_res['dists'], cmap=cmaps["viridis"], interpolation='nearest')
        # plt.plot(range(len(dtw_res["spec_idxs"])), dtw_res['aligned_sheet_idxs'], 'w-', linewidth=3, alpha=0.8)
        
        if frame_id in dtw_res["spec_idxs"]:
            col = np.where(dtw_res["spec_idxs"] == frame_id)[0][0]
            row = dtw_res['aligned_sheet_idxs'][col]
        
        dtw_res['aligned_sheet_idxs'][:col]
        plt.plot(range(col), dtw_res['aligned_sheet_idxs'][:col], '-', color=cmap[2], linewidth=5, alpha=0.8)
        
        plt.plot(col, row, 'o', markersize=10, color=cmap[2])
        
        plt.xlim([0, dtw_res['dists'].shape[1] - 1])
        plt.ylim([0, dtw_res['dists'].shape[0] - 1])
        
        plt.ylabel("Sheet", fontsize=16)
        plt.xlabel("Audio", fontsize=16)
        plt.title("Audio - Sheet - Distances", fontsize=20)
        
        plt.draw()
        plt.savefig("figs/%05d.png" % i)
