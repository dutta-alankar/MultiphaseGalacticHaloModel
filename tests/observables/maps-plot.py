# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:31:38 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib import gridspec
from itertools import product
import pickle


def plot_map(unmod: str, mod: str, ionization: str, map_type: str) -> None:
    print(unmod, mod, ionization, map_type)

    if map_type == "emission":
        cblabel = r"EM [$\times 10^{-3} \rm cm^{-6} pc$]"
    else:
        cblabel = r"DM [$\rm cm^{-3} pc$]"

    with open(
        f"figures/map_{map_type}_{unmod}_{mod}_{ionization}.pickle", "rb"
    ) as data_file:
        data = pickle.load(data_file)
        # print(list(data.keys()))
        l = data["l"]
        b = data["b"]
        map_val = data["map"]
        disk_val = data["disk"]

    l_plot = np.deg2rad(np.arange(0, 360, 45))
    b_plot = np.deg2rad(np.arange(-90, -25, 30))

    # Make plot
    levels = 100

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(1, 1)
    # Position plot in figure using gridspec.
    ax = plt.subplot(gs[0], polar=True)
    ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

    # Set x,y ticks
    plt.xticks(l_plot)  # , fontsize=8)
    plt.yticks(b_plot)  # , fontsize=8)
    ax.set_rlabel_position(18)
    # ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
    #     '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
    ax.set_yticklabels(["", "$-60^{\circ}$", "$-30^{\circ}$"])  # , fontsize=10)
    ax.set_theta_zero_location("S")
    cs = ax.contourf(
        np.deg2rad(l),
        np.deg2rad(b),
        (map_val + disk_val) * (1e3 if map_type == "emission" else 1.0),
        levels=levels,
        cmap="YlOrRd_r",
    )
    cbar = fig.colorbar(cs, pad=0.08)
    cbar.set_label(cblabel, rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.grid(color="tab:grey", linestyle="--", linewidth=1.0)
    plt.savefig(
        f"figures/map_{map_type}_{unmod}_{mod}_{ionization}.png"
    )  # , transparent=True, bbox_inches='tight')
    plt.close()

    l_mod = l - 360  # np.select([l<=180,l>180],[l,l+360])
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    cs = ax.pcolormesh(
        np.deg2rad(l_mod),
        np.deg2rad(b),
        (map_val + disk_val)
        * (1e3 if map_type == "emission" else 1.0),  # in units of 1e-3 cm^-6 pc
        cmap="inferno",
    )  # , norm=colors.LogNorm())
    cs = ax.pcolormesh(
        np.deg2rad(-l_mod),
        np.deg2rad(b),
        (map_val + disk_val) * (1e3 if map_type == "emission" else 1.0),
        cmap="inferno",
    )  # , norm=colors.LogNorm())
    ax.grid(True)
    cbar = fig.colorbar(
        cs,
        pad=0.08,
        orientation="horizontal",
        shrink=0.5,
        aspect=60,
        location="top",
        format="%.1f",
    )
    cbar.ax.tick_params(labelsize=12, length=6, width=2)
    cbar.set_label(cblabel, rotation=0, labelpad=8, fontsize=18)
    fig.tight_layout()
    plt.grid(color="tab:grey", linestyle="--", linewidth=1.0)
    plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=12)
    plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=10)
    ax.tick_params(axis="x", colors="white")

    plt.setp(ax.xaxis.get_majorticklabels())
    # Create offset transform by 5 points in x direction
    # Matplotlib figures use 72 points per inch (ppi).
    # So to to shift something by x points, you may shift it by x/72 inch.
    dx = 0 / 72.0
    dy = -30 / 72.0
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    axes_labels = ax.get_xticks().tolist()
    axes_labels = list(
        np.round(
            np.rad2deg(
                np.select(
                    [np.array(axes_labels) < 0, np.array(axes_labels) > 0],
                    [np.array(axes_labels) + 2 * np.pi, np.array(axes_labels)],
                )
            )
        )
    )
    # axes_labels = [r'$%d^{{\fontsize{50pt}{3em}\selectfont{}\circ}}$'%label for label in axes_labels]
    axes_labels = [r"$%d^{\circ}$" % label for label in axes_labels]
    ax.set_xticklabels(axes_labels)
    plt.savefig(
        f"figures/map_moll_{map_type}_{unmod}_{mod}_{ionization}.png",
        transparent=False,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]
    map_type = ["dispersion", "emission"]

    for condition in product(unmod, mod, ionization, map_type):
        plot_map(*condition)
