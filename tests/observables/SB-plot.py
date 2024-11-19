# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 01:47:55 2023

@author: alankar
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import matplotlib
import pickle
from itertools import product

## Plot Styling
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
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
matplotlib.rcParams["lines.linewidth"] = 1.5
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.minor.width"] = 0.45
matplotlib.rcParams["xtick.minor.width"] = 0.45
matplotlib.rcParams["ytick.major.size"] = 4.0
matplotlib.rcParams["xtick.major.size"] = 4.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = 10.0
matplotlib.rcParams["xtick.minor.pad"] = 10.0
matplotlib.rcParams["ytick.major.pad"] = 6.0
matplotlib.rcParams["ytick.minor.pad"] = 6.0
matplotlib.rcParams["xtick.labelsize"] = 24.0
matplotlib.rcParams["ytick.labelsize"] = 24.0
matplotlib.rcParams["axes.titlesize"] = 24.0
matplotlib.rcParams["axes.labelsize"] = 28.0
matplotlib.rcParams["axes.labelpad"] = 8.0
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True
# plt.style.use('dark_background')

def calc_SB(
    Emin: float, Emax: float, energy: np.ndarray, sed: np.ndarray, verbose: bool = True,
) -> float:
    # pc = 3.0856775807e18
    # kpc = 1e3 * pc
    select = np.logical_and(energy >= Emin, energy <= Emax)
    SB = np.trapz(sed[select], energy[select])
    if verbose:
        print(
              f"SB ({Emin:.1f}-{Emax:.1f} keV): {(SB):.6e} erg cm^-2 s^-1 deg^-2"
        )
    return SB


def spectrum(
    unmod: str, mod: str, ionization: str, fig: plt.figure
) -> matplotlib.lines.Line2D:
    print(unmod, mod, ionization)

    with open(f"SB_{unmod}_{mod}_{ionization}.pickle", "rb") as data_file:
        data = pickle.load(data_file)
        energy = np.array(data["energy"])
        sb_hot = np.array(data["sb_hot"])
        sb_warm = np.array(data["sb_warm"])
        sb_disk = np.array(data["sb_disk"])
        rCGM = data["rCGM"]
    if (unmod == 'isoth' and mod == 'isochor'):   
         colour = 'teal'
    elif (unmod == 'isoth' and mod == 'isobar'):   
         colour = 'gold'
    elif (unmod == 'isent' and mod == 'isochor'):   
         colour = 'coral'
    elif (unmod == 'isent' and mod == 'isobar'):   
         colour = 'silver'               
    else :     
         print('error')
         sys.exit(1)
    
    sb_tot = sb_hot + sb_warm + sb_disk
    plot_quan = sb_tot
    
    Ebands = [[0.3, 0.6], [0.6, 2.0]]  # keV
    print("Total")
    _ = list(map(lambda till: calc_SB(*till, energy, sb_tot), Ebands))
    print("Disk")
    _ = list(map(lambda till: calc_SB(*till, energy, sb_disk), Ebands))
    print("Hot")
    _ = list(map(lambda till: calc_SB(*till, energy, sb_hot), Ebands))
    print("Warm")
    _ = list(map(lambda till: calc_SB(*till, energy, sb_warm), Ebands))
    
    bands = [calc_SB(*Eband, energy, plot_quan, verbose=False) for Eband in Ebands]
    ax = fig.gca()
    unmod_label = (
        r"$\gamma = 1$" if unmod == "isoth" else r"$\gamma = 5/3$"
    ) + " polytrope"
    mod_label = "(IC)" if mod == "isochor" else "(IB)"
    
    plotted_line = ax.loglog(
        energy,
        plot_quan,
        color= colour,
        alpha = 0.5,
        label=f"{unmod_label} {mod_label}",
    )
    return (plotted_line[0], bands)


if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 10))
    unmod = ["isoth","isent"]
    mod = ["isochor",]# "isobar"]
    ionization = ["PIE",]# "CIE"]

    plt.gca().axvspan(0.3, 0.6, alpha=0.5, color="darkkhaki")     
    plt.gca().axvspan(0.6, 2.0, alpha=0.35, color="rebeccapurple")

    curves = []
    band_vals = []
    for condition in product(unmod, mod, ionization):
        tmp = spectrum(*condition, fig)
        curves.append(tmp[0])
        band_vals.append(tmp[1])

    curves[-2].set_alpha(0.9) # ('isoth', 'isochor', 'PIE')
    curves[-1].set_alpha(0.8) # ('isent', 'isochor', 'PIE')

    # isent
    plt.hlines(band_vals[-1][0], 0.3, 0.6, colors = 'coral', linestyle='-', linewidth=4.0, zorder=50)
    plt.hlines(band_vals[-1][1], 0.6, 2.0, colors = 'coral', linestyle='-', linewidth=4.0, zorder=50)
    # isoth
    plt.hlines(band_vals[-2][0], 0.3, 0.6, colors = 'teal', linestyle=':', linewidth=4.0, zorder=50)
    plt.hlines(band_vals[-1][1], 0.6, 2.0, colors = 'teal', linestyle=':', linewidth=4.0, zorder=50)
    # band observed; arrows inserted here
    plt.plot(0.25,15.6e-13, color = "darkkhaki", 
             marker=r'$\rightarrow$', markersize=35,
             zorder=50, alpha=0.9, markeredgewidth=0.0)
    plt.plot(2.35,4.9e-13, color = "rebeccapurple", 
             marker=r'$\leftarrow$', markersize=35,
             zorder=50, alpha=0.9, markeredgewidth=0.0)
    
    plt.ylim(ymin=9e-17, ymax=9e-8)
    plt.ylabel("Surface brightness", size=28, labelpad=35)
    plt.xlabel(r"Energy", size=28, horizontalalignment='center')
    plt.gca().xaxis.set_label_coords(0.45, -0.06)
    # Inserting text rotated default
    plt.text(2.05e-3, 1.0e-14, r"[$erg\ s^{-1}\ cm^{-2}\ deg^{-2}\ keV^{-1}$]", 
            rotation=90, rotation_mode='anchor', fontsize=22)
    plt.text(2.9e-1, 1.35e-17, r"[$keV$]", 
            rotation=0, rotation_mode='anchor', fontsize=22)
    plt.xlim(xmin=5e-3, xmax=1.2e1)
    plt.legend(
        loc="lower left", prop={"size": 20}, 
        framealpha=0.5, shadow=False, fancybox=True,
        bbox_to_anchor=[0.02, 0.12],
    )
    plt.tick_params(axis="both", which="major", length=15, width=1.5, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=8, width=1, labelsize=22)
    # plt.gca().yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    plt.gca().yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

    '''
    plt.savefig(
        f"figures/SB_{unmod[0]}+{unmod[1]}_{mod[0]}_{ionization[0]}.png",
        transparent=False,
    )
    '''  
    plt.tight_layout()
    plt.savefig(
        f"figures/SB.png",
        transparent=False,
        dpi=200,
    )
    plt.show()
