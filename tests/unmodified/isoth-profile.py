# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:53:17 2022

@author: Alankar
"""

import sys

sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
from misc.constants import kpc, kB, yr
from unmodified import isoth
import numpy as np
import matplotlib.pyplot as plt
from misc.coolLambda import cooling_approx

sig = 0.3
Temperature = 1.5e6 * np.exp(-(sig**2) / 2)

Npts = 200
Model = isoth.IsothermalUnmodified(THot=Temperature, ionization="CIE")
radius = np.linspace(9.0, 250, Npts)  # kpc
rho, prsTh, _, prsTurb, prsTot, nH, mu = Model.ProfileGen(radius)
metallicity = Model.metallicity

n = Model.ndens
nH = Model.nH  # CGS

isobaric = 0


def tcool(ndens, nH, Temp, met):
    return (1.5 + isobaric) * ndens * kB * Temp / (nH * nH * cooling_approx(Temp, met))


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(radius, tcool(n, nH, Temperature, metallicity) / (1e9 * yr))
plt.grid()
plt.ylabel(r"tcool, hot [Gyr]")
plt.xlabel("radius [kpc]")
ax.yaxis.set_ticks_position("both")
ax.set_xlim(0, 250)
plt.savefig("figures/tcool-isoth.png")
plt.cla()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.log10(nH), mu)
plt.grid()
plt.ylabel(r"$\mu$ ($n_H$)")
plt.xlabel(r"$n_H$ [$cm^{-3}$]")
ax.yaxis.set_ticks_position("both")
plt.savefig("figures/mu-isoth.png")
plt.cla()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, prsTot / kB)
plt.semilogy(
    [
        Model.Halo.r0 * Model.Halo.UNIT_LENGTH / kpc,
        250,
    ],
    [
        4580,
        230,
    ],
    "o",
    color="black",
)
plt.grid()
plt.ylabel(r"$\rm <P_{hot}(r)>/k_B$ [CGS]")
plt.xlabel("radius [kpc]")
ax.yaxis.set_ticks_position("both")
ax.set_xlim(0, 250)
ax.set_ylim(50, 1e4)
plt.savefig("figures/pres-isoth.png")
plt.cla()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, nH, label=r"$n_H$")
plt.semilogy(radius, n, label=r"$n$")
plt.semilogy([50, 100], [0.83e-4, 1.3e-4], "o", color="black")
plt.grid()
plt.ylabel(r"$\rm <n_{dens}(r)>$ [CGS]")
plt.xlabel("radius [kpc]")
ax.yaxis.set_ticks_position("both")
ax.set_xlim(0, 250)
ax.set_ylim(1e-6, 1e-2)
plt.legend(loc="best")
plt.savefig("figures/nDens-isoth.png")
plt.cla()
