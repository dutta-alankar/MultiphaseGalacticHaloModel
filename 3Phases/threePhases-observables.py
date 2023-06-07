# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:42:10 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from scipy.optimize import root

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
sys.path.append("../tests/observables")
from astro_plasma import Ionization
from misc.HaloModel import HaloModel
from parse_observation import observedColDens
from misc.constants import mp, mH, kpc, Xp

N_pdf = lambda x, mu, sig: (1.0 / (np.sqrt(2 * np.pi) * sig)) * np.exp(
    -(((x - mu) / (np.sqrt(2) * sig)) ** 2)
)

mode = "PIE"
Z0 = 1.0
ZrCGM = 0.3
p = Z0 / ZrCGM
metallicity = (
    1.5
    * (p - (p**2 - 1) * np.arcsin(1.0 / np.sqrt(p**2 - 1)))
    * Z0
    * np.sqrt(p**2 - 1)
)
redshift = 0.2

M200 = 1e12  # MSun
halo = HaloModel(M200=M200)
rCGM = 1.1 * halo.r200 * (halo.UNIT_LENGTH / kpc)  # kpc
r200 = halo.r200 * (halo.UNIT_LENGTH / kpc)  # kpc
PbykB = 1e1

f_Vh, f_Vw, f_Vc, x_h, x_w, x_c, sig_h, sig_w, sig_c, T_u = np.load(
    "./figures/mcmc_opt-parameters.npy"
)

Temperature = np.logspace(3.8, 6.8, 50)
x = np.log(Temperature / T_u)

V_pdf_fv = (
    f_Vh**2 * N_pdf(x, x_h, sig_h)
    + f_Vw**2 * N_pdf(x, x_w, sig_w)
    + f_Vc**2 * N_pdf(x, x_c, sig_c)
)
# This volume fraction square is because global quantities come up calculation of column density
# On the other hand, P = n kB T , is where n is local
# One volume fraction comes from the pdf while the other is from local to global density conversion

mu = Ionization.interpolate_mu

nH_guess = (
    (PbykB * (6e5 / Temperature) / Temperature) * (Xp(metallicity) * 0.61) / (mH / mp)
)  # guess
# This is a local quantity
nH = 10.0 ** np.array(
    [
        root(
            lambda LognH: PbykB
            * (6e5 / Temperature[i])
            * Xp(metallicity)
            * mu(
                10.0**LognH,
                Temperature[i],
                metallicity,
                redshift,
                mode,
            )
            - 10.0**LognH * Temperature[i] * (mH / mp),
            np.log10(nH_guess[i]),
        ).x[0]
        for i in range(Temperature.shape[0])
    ]
)

"""
# Used during debugging
mu_val = np.array([mu(nH[i], Temperature[i], metallicity, redshift, mode) for i in range(Temperature.shape[0])])
print(nH*(mH/mp)*Temperature/mu_val)
print(nH)
"""


def nIon_global_avg(element):
    nIon = Ionization.interpolate_num_dens
    nIon_local = np.array(
        [
            nIon(nH[i], T_val, metallicity, redshift, mode, element=element)
            for i, T_val in enumerate(Temperature)
        ]
    )
    nIon_g_avg = np.trapz(nIon_local * V_pdf_fv, x)
    return nIon_g_avg


num_dens = Ionization.interpolate_num_dens

ne_local = np.array(
    [
        num_dens(nH[i], T_val, metallicity, redshift, mode, "electron")
        for i, T_val in enumerate(Temperature)
    ]
)
ni_local = np.array(
    [
        num_dens(nH[i], T_val, metallicity, redshift, mode, "ion")
        for i, T_val in enumerate(Temperature)
    ]
)


ne_global_avg = np.trapz(ne_local * V_pdf_fv, x)
ni_global_avg = np.trapz(ni_local * V_pdf_fv, x)

b = np.linspace(9.0, 1.05 * r200, 200)  # kpc
column_length = 2 * np.sqrt(rCGM**2 - b**2)

os.makedirs("./figures/observations/", exist_ok=True)


def IonColumn(element, ylim=None, fignum=None, color=None):
    if fignum is None:
        plt.figure(figsize=(13, 10))
    else:
        plt.figure(num=fignum, figsize=(13, 10))
    if color == None:
        color = "tab:blue"

    observation = observedColDens()

    NIon = np.nan_to_num(
        nIon_global_avg("".join(element.split())) * column_length * kpc
    )  # cm^-2
    plt.plot(
        b / r200,
        NIon,
        color=color,
        label=r"$\rm N_{%s, %s}$" % ("".join(element.split()), mode),
        linewidth=5,
    )

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
        color=color,
        label=r"$\rm N_{%s, obs}$" % ("".join(element.split()),),
        markersize=12,
    )
    plt.plot(
        impact_select_min / rvir_select_min,
        10.0**coldens_min,
        "^",
        color=color,
        markersize=12,
    )
    plt.plot(
        impact_select_max / rvir_select_max,
        10.0**coldens_max,
        "v",
        color=color,
        markersize=12,
    )

    plt.xscale("log")
    plt.yscale("log")
    if ylim != None:
        plt.ylim(*ylim)
    plt.xlim(6e-2, 1.2)
    plt.xlabel(r"$b/R_{vir}$", size=28)
    plt.ylabel(r"Column density $[cm^{-2}]$", size=28)

    plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
    plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
    plt.grid()
    plt.tight_layout()
    # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    # leg.set_title("Column density predicted by three phase model",prop={'size':20})
    if fignum == None:
        plt.legend(loc="upper right", ncol=1, fancybox=True, fontsize=25)
        plt.savefig(
            "./figures/observations/N_%s-3p.png" % ("".join(element.split()),),
            transparent=False,
        )
        # plt.show()
        plt.close()


# --------------------- DM and EM ------------------
DM = ne_global_avg * column_length * 1e3  # cm^-3 pc
EM = ne_global_avg * ni_global_avg * column_length * 1e3  # cm^-6 pc

# DM
plt.figure(figsize=(13, 10))
plt.plot(b / r200, DM, color="firebrick", linewidth=5)
plt.xscale("log")
plt.yscale("log")
plt.ylim(4.0, 60.0)
plt.xlim(6e-2, 1.2)
plt.xlabel(r"$b/R_{vir}$", size=28)
plt.ylabel(r"DM [$cm^{-3} pc$]", size=28)
plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
# set the linewidth of each legend object
# for legobj in leg.legendHandles:
# leg.set_title("Column density predicted by three phase model",prop={'size':20})
plt.savefig("./figures/observations/DM-3p.png", transparent=False)
# plt.show()
plt.close()

# EM
plt.figure(figsize=(13, 10))
plt.plot(b / r200, EM * 1e3, color="firebrick", linewidth=5)
plt.xscale("log")
plt.yscale("log")
# plt.ylim(2e-2, 0.3)
# plt.xlim(6e-2, 1.2)
plt.xlabel(r"$b/R_{vir}$", size=28)
plt.ylabel(r"EM [$\times 10^{-3} cm^{-6} pc$]", size=28)
# leg = plt.legend(loc='lower left', ncol=1, fancybox=True, fontsize=25)
plt.tick_params(axis="both", which="major", length=12, width=3, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
# set the linewidth of each legend object
# for legobj in leg.legendHandles:
# leg.set_title("Column density predicted by three phase model",prop={'size':20})
plt.savefig("./figures/observations/EM-3p.png", transparent=False)
# plt.show()
plt.close()

# ---------- Individual Ions -------------------------
element = "O VI"
IonColumn(element, ylim=(10**14.0, 10.0**15.3))

element = "Mg II"
IonColumn(element)

element = "Si IV"
IonColumn(element)

element = "S III"
IonColumn(element)

element = "N V"
IonColumn(element)

element = "C III"
IonColumn(element)

element = "C II"
IonColumn(element)

# all together
fignum = 100
fig = plt.figure(figsize=(13, 10), num=fignum)
# print('Figures: ', plt.get_fignums())
element = "O VI"
IonColumn(element, fignum=fignum, color="coral")

element = "Mg II"
IonColumn(element, fignum=fignum, color="lightseagreen")

fig = plt.figure(num=fignum)
leg = plt.legend(loc="upper right", ncol=2, fancybox=True, fontsize=25)
plt.ylim(2e10, 2e16)
plt.grid()
plt.savefig("./figures/observations/N_OVI+MgII-3p.png", transparent=False)
# plt.show()
# plt.close()
