# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:28:43 2023

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
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


def plot_column_density(unmod, mod, ion, ax, colors):
    states = ["PIE", "CIE"]

    for indx, ionization in enumerate(states):
        with open(
            f"figures/N_{ion}_{unmod}_{mod}_{ionization}.pickle", "rb"
        ) as data_file:
            data = pickle.load(data_file)
            impact = data["impact"]
            column_density = data[f"N_{ion}"]
            rCGM = data["rCGM"]
            ax.loglog(
                np.array(impact) / rCGM,
                column_density,
                linestyle="-",
                color=colors[indx],
            )


if __name__ == "__main__":
    unmod = "isent"
    mod = "isochor"
    element = "N V"

    plt.figure(figsize=(13, 10))
    ax = plt.gca()
    colors = ["cadetblue", "gold"]

    plot_column_density(unmod, mod, "".join(element.split()), ax, colors=colors)

    observation = observedColDens()

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
    if len(e_coldens_detect != 0):
        yerr = np.log(10) * e_coldens_detect * 10.0**coldens_detect
        plt.errorbar(
            impact_select_detect / rvir_select_detect,
            10.0**coldens_detect,
            yerr=yerr,
            fmt="o",
            color="black",
            label=r"Observations",
            markersize=10,
        )
        plt.plot(
            impact_select_min / rvir_select_min,
            10.0**coldens_min,
            "^",
            color="black",
            label=r"Observations",
            markersize=10,
        )
        plt.plot(
            impact_select_max / rvir_select_max,
            10.0**coldens_max,
            "v",
            color="black",
            label=r"Observations",
            markersize=10,
        )
    obs_handles, obs_labels = plt.gca().get_legend_handles_labels()

    line_pie = matplotlib.lines.Line2D(
        [0],
        [0],
        color=colors[0],
        linestyle="-",
        linewidth=4.0,
        label="PIE",
    )
    line_cie = matplotlib.lines.Line2D(
        [0],
        [0],
        color=colors[1],
        linestyle="-",
        linewidth=4.0,
        label="CIE",
    )
    legend = plt.legend(
        loc="lower left",
        prop={"size": 20},
        framealpha=0.3,
        shadow=False,
        fancybox=True,
        bbox_to_anchor=(0.03, 0.01),
        fontsize=18,
        ncol=1,
        handles=[
            line_pie,
            line_cie,
            tuple(obs_handles[::-1]),
        ],
        labels=[
            r"PIE",
            r"CIE",
            obs_labels[0],
        ],
        handler_map={
            tuple: HandlerTuple(ndivide=None),
        },
        handlelength=1.5,
    )
    legend.get_frame().set_edgecolor("rebeccapurple")
    legend.get_frame().set_facecolor("ivory")
    legend.get_frame().set_linewidth(1.0)
    ax.add_artist(legend)

    plt.xlabel(r"Impact parameter [$r_{vir}$]", size=28)
    plt.ylabel(r"Column density of %s ($cm^{-2}$)" % element, size=28)
    plt.tick_params(axis="both", which="major", length=10, width=2, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=6, width=1, labelsize=22)
    plt.xlim(xmin=0.05, xmax=1.1)

    plt.title(
        r"$\gamma = %s$ polytrope (%s)"
        % ("1" if unmod == "isoth" else "5/3", "IC" if mod == "isochor" else "IB"),
        size=28,
    )

    plt.savefig(f"figures/column_density_{element}_PIEvsCIE.png", transparent=False)
