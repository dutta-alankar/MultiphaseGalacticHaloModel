# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:31:38 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib import gridspec
from matplotlib import colors
from itertools import product
import pickle


def plot_map(unmod: str, mod: str, ionization: str, map_type: str) -> None:
    print(unmod, mod, ionization, map_type)

    if unmod == "isoth":
        title = r"$\gamma = 1$ polytrope"
    else:
        title = r"$\gamma = 5/3$ polytrope"
    if mod == "isochor":
        title += " (IC)"
    else:
        title += " (IB)"
    cmap = "nipy_spectral_r" 
    if map_type == "emission":
        cblabel = r"%sEM (Halo/Disk)" % (title + "\n",)
        vmin = 0.0
        vmax = 1.5
    else:
        cblabel = r"%sDM (Halo/Disk)" % (title + "\n",)
        vmin = 0.0
        vmax = 4.0

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
    b_plot = np.deg2rad(np.arange(-90, 90, 30))

    # Make plot
    levels = 200
    '''
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(1, 1)
    # Position plot in figure using gridspec.
    ax = plt.subplot(gs[0], polar=True)
    ax.set_ylim(np.deg2rad(-90), np.deg2rad(90))

    # Set x,y ticks
    plt.xticks(l_plot , fontsize=18)
    plt.yticks(b_plot , fontsize=18)
    ax.set_rlabel_position(18)
    # ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
    #     '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
    ax.set_yticklabels(["", "$-60^{\circ}$", "$-30^{\circ}$","$0^{\circ}$","$30^{\circ}$","$60^{\circ}$"]   , fontsize=18)
    ax.set_theta_zero_location("S")
    cs = ax.contourf(
        np.deg2rad(l),
        np.deg2rad(b),
        map_val / disk_val ,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(cs, pad=0.08)
    cbar.set_label(cblabel, rotation=270, labelpad=18, fontsize=18)
    # cbar.mappable.set_clim(vmin, vmax)
    fig.tight_layout()
    plt.grid(color="tab:grey", linestyle="--", linewidth=1.0)
    plt.title(title, size=20)
    
    #plt.show()
    
    plt.savefig(
        f"figures/ratio_map_{map_type}_{unmod}_{mod}_{ionization}.png",
        transparent=False,
        bbox_inches="tight",
    )
    plt.close()
    '''
    l_mod = l - 360  # np.select([l<=180,l>180],[l,l+360])
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    if (map_type == "emission" or "dispersion"):
      cs = ax.pcolormesh(
        np.deg2rad(l_mod),
        np.deg2rad(b),
        map_val/ disk_val,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)
        #norm=colors.LogNorm())
      cs = ax.pcolormesh(
        np.deg2rad(-l_mod),
        np.deg2rad(b),
        map_val/ disk_val,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)
        #norm=colors.LogNorm())
      #plt.plot(np.deg2rad(230), np.deg2rad(30), markerstyle = '*',color = 'white')  
    else :
      cs = ax.pcolormesh(
        np.deg2rad(l_mod),
        np.deg2rad(b),
        map_val/ disk_val ,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)
      cs = ax.pcolormesh(
        np.deg2rad(-l_mod),
        np.deg2rad(b),
        map_val/ disk_val ,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)
    ax.grid(True)
    cbar = fig.colorbar(
        cs,
        pad=0.08,
        orientation="horizontal",
        shrink=0.6,
        aspect=70,
        location="top",
        format="%2.2f",
    )
    # cs.set_clim(vmin, vmax)
    if map_type == "emission":
       ax.scatter([np.deg2rad(230-360),],[np.deg2rad(30.),],marker='X',c='black', s=90,linewidths=0.3)
    if map_type == "dispersion":
       ax.scatter([np.deg2rad(142),],[np.deg2rad(41.),],marker='X',c='black', s=90,linewidths=0.3)  
    cbar.ax.tick_params(labelsize=14, length=6, width=3)
    cbar.set_label(cblabel, rotation=0, labelpad=8, fontsize=18)
    # cbar.mappable.set_clim(vmin, vmax)
    fig.tight_layout()
    plt.grid(color="tab:grey", linestyle="--", linewidth=1.0)
    plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=12)
    plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=10)
    ax.tick_params(axis="x", colors="black")

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
    # if map_type == "dispersion":
    #     plt.title(title, size=20)
    
    #plt.show()
    
    plt.savefig(
        f"figures/ratio_map_moll_{map_type}_{unmod}_{mod}_{ionization}.png",
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
