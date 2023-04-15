# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:24:09 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
from itertools import product
from parse_observation import observedColDens

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
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True


def plot_column_density(unmod, mod, ion):
    states = ["PIE", "CIE"]

    for ionization in states:
        with open(
            f"figures/N_{ion}_{unmod}_{mod}_{ionization}.pickle", "rb"
        ) as data_file:
            data = pickle.load(data_file)
            impact = data["impact"]
            column_density = data[f"N_{ion}"]
            rCGM = data["rCGM"]

            extra_label = ("Isothermal" if unmod == "isoth" else "Isentropic") + (
                " (IC)" if mod == "isochor" else "(IB)"
            )
            if ionization == "PIE":
                pl1 = plt.loglog(
                    np.array(impact) / rCGM,
                    column_density,
                    label=r"$N_{%s}$ " % (ion,) + extra_label,
                    linestyle="-",
                )
            else:
                plt.loglog(
                    np.array(impact) / rCGM,
                    column_density,
                    linestyle="--",
                    color=pl1[0].get_color(),
                )


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ion = ["NV"]  # , "NV"]

    plt.figure(figsize=(13, 10))

    for condition in product(unmod, mod, ion):
        plot_column_density(*condition)

    observation = observedColDens()

    element = "N V"
    (
        gal_id_min,
        gal_id_max,
        gal_id_detect,
        rvir_select_min,
        rvir_select_max,
        rvir_select_detect,
        impact_select_min,
        impact_select_max,
        impact_select_detect,
        coldens_min,
        coldens_max,
        coldens_detect,
        e_coldens_detect,
    ) = observation.col_density_gen(element=element)

    yerr = np.log(10) * e_coldens_detect * 10.0**coldens_detect
    plt.errorbar(
        impact_select_detect / rvir_select_detect,
        10.0**coldens_detect,
        yerr=yerr,
        fmt="o",
        color="black",
        label=r"$\rm N_{%s, obs}$" % element,
        markersize=10,
    )
    plt.plot(
        impact_select_min / rvir_select_min,
        10.0**coldens_min,
        "^",
        color="black",
        markersize=10,
    )
    plt.plot(
        impact_select_max / rvir_select_max,
        10.0**coldens_max,
        "v",
        color="black",
        markersize=10,
    )

    plt.legend()
    plt.xlabel(r"Impact parameter [$r_{vir}$]")
    plt.ylabel(r"Column density ($cm^{-2}$)")
    plt.xlim(xmin=0.05, xmax=1.1)
    plt.ylim(ymin=10**11.7, ymax=10.0**15.3)
    plt.savefig(f"figures/column_density{ion}.png", transparent=False)
