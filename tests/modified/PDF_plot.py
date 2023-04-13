#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 08:30:44 2023

@author: alankar
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from itertools import product

## Plot Styling
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["legend.handletextpad"] = 0.4
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["lines.linewidth"] = 3.0
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.minor.width"] = 0.45
matplotlib.rcParams["xtick.minor.width"] = 0.45
matplotlib.rcParams["ytick.major.size"] = 4.0
matplotlib.rcParams["xtick.major.size"] = 4.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = 6.0
matplotlib.rcParams["xtick.minor.pad"] = 6.0
matplotlib.rcParams["ytick.major.pad"] = 6.0
matplotlib.rcParams["ytick.minor.pad"] = 6.0
matplotlib.rcParams["xtick.labelsize"] = 24.0
matplotlib.rcParams["ytick.labelsize"] = 24.0
matplotlib.rcParams["axes.titlesize"] = 24.0
matplotlib.rcParams["axes.labelsize"] = 28.0
plt.rcParams['font.size'] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True


def plot_profile(radius, unmod, all_mods, ionization, figure):
    fig = figure
    ax = fig.gca()
    for mod in all_mods:
        with open(f'figures/{unmod}_{mod}_{ionization}_distrib-r={radius:.1f}kpc.pickle','rb') as data_file:
            profile = pickle.load(data_file)
            Tcutoff = profile["T_cutoff"]
            cutoff = profile["cutoff"]
            TmedVu = profile["T_hot_M"]*np.exp(profile["sig_u"]**2/2.)
            TmedVW = profile["T_med_VW"]
            TempDist = profile["TempDist"]
            gvH_mod = profile["Hot_mod"]
            gvh = profile["gv_h"]
            gvw = profile["gv_w"]
            if mod=="isochor":
                ax.vlines(np.log10(Tcutoff), 1e-3, 2.1, colors='black', linestyles='--', label=r'$T_c\ (t_{\rm cool}/t_{\rm ff}=%.1f)$'%cutoff, 
                             linewidth=3, zorder=20, alpha=0.6)
                ax.vlines(np.log10(TmedVu), 1e-3, 2.1, colors='tab:red', linestyles=':', label=r'$T_{med,V}^{(h)}$', 
                             linewidth=3, zorder=30)
                ax.vlines(np.log10(TmedVW), 1e-3, 2.1, colors='tab:blue', linestyles=':', label=r'$T_{med,V}^{(w)}$', 
                             linewidth=3, zorder=40)
                ax.semilogy(np.log10(TempDist), 
                             gvH_mod, 
                             color='tab:red', label='hot, modified', 
                             linewidth=5, zorder=5)
                ax.semilogy(np.log10(TempDist), gvh, color='tab:red', alpha=0.5, label='hot, unmodified', 
                             linewidth=5, zorder=6)
            ax.semilogy(np.log10(TempDist), gvw, color='tab:blue' if mod=="isobar" else "tab:cyan", 
                         label=f'warm, {mod}', linestyle='--', 
                         linewidth=5, zorder=7)

if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    all_mods = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]
    
    for condition in product(unmod, ionization):
        radius = 200.0 if unmod=="isoth" else 20.0 # kpc
        fig = plt.figure(figsize=(13,10))
        plot_profile(radius=radius, unmod=condition[0], all_mods=all_mods, ionization=condition[1], figure=fig)
        plt.title(r'Modified probability distribution at $r = %.1f$ kpc (%s)'%(radius, condition[1]), size=28)
        plt.ylim(1e-3, 2.1)
        plt.xlim(5, 7)
        plt.ylabel(r'$T \mathscr{P}_V(T)$')
        plt.xlabel(r'$\log_{10} (T [K])$')
        # ax.yaxis.set_ticks_position('both')
        # plt.tick_params(axis='both', which='major', labelsize=15, direction="out", pad=5)
        # plt.tick_params(axis='both', which='minor', labelsize=15, direction="out", pad=5)
        plt.legend(loc='upper right', prop={'size': 20}, framealpha=0.3, shadow=False, fancybox=True, bbox_to_anchor=(1.1, 1))
        plt.savefig(f'figures/PDF_{condition[0]}_{condition[1]}.png', transparent=False)
        # plt.show()
