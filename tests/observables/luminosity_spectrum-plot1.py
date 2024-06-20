# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 01:47:55 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


def calc_SB(
    Emin: float, Emax: float, energy: np.ndarray, sed: np.ndarray, rCGM: float
) -> None:
    pc = 3.0856775807e18
    kpc = 1e3 * pc

    select = np.logical_and(energy >= Emin, energy <= Emax)
    SB = np.trapz(sed[select], energy[select]) / (4 * (np.pi * rCGM * kpc) ** 2)
    print(
        f"SB ({Emin:.1f}-{Emax:.1f} keV): {(SB/(180/np.pi)):.2e} erg cm^-2 s^-1 deg^-2"
    )


def spectrum(
    unmod: str, mod: str, ionization: str, fig: plt.figure
) -> matplotlib.lines.Line2D:
    print(unmod, mod, ionization)

    with open(f"figures/spectrum1_{unmod}_{mod}_{ionization}.pickle", "rb") as data_file:
        data = pickle.load(data_file)
        energy = np.array(data["energy"])
        luminosity = np.array(data["luminosity"])
        rCGM = data["rCGM"]

    Emin = 0.3  # keV
    Emax = [0.6, 2.0]  # keV
    _ = list(map(lambda till: calc_SB(Emin, till, energy, luminosity, rCGM), Emax))

    ax = fig.gca()
    unmod_label = (
        r"$\gamma = 1$" if unmod == "isoth" else r"$\gamma = 5/3$"
    ) + " polytrope"
    mod_label = "(IC)" if mod == "isochor" else "(IB)"
    plotted_line = ax.loglog(
        energy,
        luminosity,
        color="teal" if unmod == "isoth" else "coral",
        label=f"{unmod_label} {mod_label}",
    )
    return plotted_line[0]


if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 10))
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]

    plt.gca().axvspan(0.3, 0.6, alpha=0.3, color="rebeccapurple")
    plt.gca().axvspan(0.3, 2.0, alpha=0.3, color="khaki")

    curves = []
    for condition in product(
        unmod,
        [
            mod[0],
        ],
        [
            ionization[0],
        ],
    ):
        curves.append(spectrum(*condition, fig))

    curves[-2].set_alpha(0.8)
    curves[-1].set_alpha(0.7)
    plt.ylim(ymin=1e30, ymax=1e46)
    plt.ylabel(r"Luminosity [$erg\ s^{-1}\ keV^{-1}$]", size=28, color="black")
    plt.xlabel(r"E [keV]", size=28, color="black")
    plt.xlim(xmin=5e-3, xmax=1.2e1)
    plt.legend(
        loc="lower left", prop={"size": 20}, framealpha=0.3, shadow=False, fancybox=True
    )
    plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
    plt.savefig(
        f"figures/spectrum_{unmod[0]}+{unmod[1]}_{mod[0]}_{ionization[0]}.png",
        transparent=False,
    )
    # plt.show()
