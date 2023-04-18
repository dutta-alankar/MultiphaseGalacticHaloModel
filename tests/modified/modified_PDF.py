# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:15:09 2022

@author: Alankar
"""

import sys

sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import product
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isobarcool import IsobarCoolRedistribution
from modified.isochorcool import IsochorCoolRedistribution

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
matplotlib.rcParams["legend.handlelength"] = 2
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True


radius = 20.0  # kpc
TmedVW = 3.0e5
sig = 0.3
cutoff = 4.0
redshift = 0.2


def gen_PDF(unmod, ionization):
    print(unmod, ionization)
    fig = plt.figure(figsize=(13, 10))

    # ------ Unmodified -------
    if unmod == "isoth":
        TmedVH = 1.5e6
        THotM = TmedVH * np.exp(-(sig**2) / 2)

        unmodified = IsothermalUnmodified(
            THot=THotM,
            P0Tot=4580,
            alpha=1.9,
            sigmaTurb=60,
            M200=1e12,
            MBH=2.6e6,
            Mblg=6e10,
            rd=3.0,
            r0=8.5,
            C=12,
            redshift=redshift,
            ionization=ionization,
        )
    else:
        nHrCGM = 1.1e-5
        TthrCGM = 2.4e5
        sigmaTurb = 60
        ZrCGM = 0.3
        unmodified = IsentropicUnmodified(
            nHrCGM=nHrCGM,
            TthrCGM=TthrCGM,
            sigmaTurb=sigmaTurb,
            ZrCGM=ZrCGM,
            redshift=redshift,
            ionization=ionization,
        )

    # -------------- Isobar -------------
    mod_isobar = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff, isobaric=0)
    mod_isobar.PlotDistributionGen(radius, figure=fig)

    # -------------- Isochor -------------
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    mod_isochor.PlotDistributionGen(radius, figure=fig)

    plt.title(
        r"$r = $%.1f kpc [isothermal with isobaric and isochoric modifications] (%s)"
        % (radius, ionization),
        size=20,
    )
    plt.ylim(1e-3, 2.1)
    plt.xlim(5, 7)
    plt.ylabel(r"$T \mathscr{P}_V(T)$", size=28)
    plt.xlabel(r"$\log_{10} (T [K])$", size=28)
    # ax.yaxis.set_ticks_position('both')
    plt.tick_params(axis="both", which="major", labelsize=15, direction="out", pad=5)
    plt.tick_params(axis="both", which="minor", labelsize=15, direction="out", pad=5)
    plt.legend(
        loc="upper right",
        prop={"size": 20},
        framealpha=0.3,
        shadow=False,
        fancybox=True,
        bbox_to_anchor=(1.1, 1),
    )
    plt.savefig(f"figures/{unmod}_modified_PDF_{ionization}.png", transparent=False)
    # plt.show()


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    ionization = ["PIE", "CIE"]

    for condition in product(unmod, ionization):
        gen_PDF(*condition)
