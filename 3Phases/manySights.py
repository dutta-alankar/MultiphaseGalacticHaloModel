# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:15:04 2023

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
import numpy as np
from typing import Union
from scipy.interpolate import interp1d
from scipy.optimize import root
from misc.constants import kpc, kB, mH, mp, Xp
from astro_plasma import Ionization
from unmodified.isoth import IsothermalUnmodified
import matplotlib.pyplot as plt

np.random.seed(10)

element = "MgII"

mu = Ionization.interpolate_mu
nIon = Ionization.interpolate_num_dens

mode = "PIE"
redshift = 0.2

# -----------------------------------------
TmedVH = pow(10.0, 5.636)
TmedVW = pow(10.0, 4.101)
sig = 0.3
THotM = TmedVH * np.exp(-(sig**2) / 2)

radius = np.linspace(9.0, 250, 30)  # kpc

# PIE
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
    ionization=mode,
)
rho, prsTh, prsnTh, prsTurb, prsTot, nH, mu_val = unmodified.ProfileGen(radius)


prs = interp1d(radius, prsTh, fill_value="extrapolate")  # CGS
density = interp1d(radius, rho, fill_value="extrapolate")  # CGS
metallicity = interp1d(radius, unmodified.metallicity, fill_value="extrapolate")

print("Hot phase calculation complete!")


def intersect_clouds(clouds, cloud_size, los, rCGM):
    b = los["b"]
    phi = los["phi"]

    los_x = b * np.cos(phi)
    los_y = b * np.sin(phi)

    r1 = np.array([los_x, los_y, 0.0])
    r2 = np.array([los_x, los_y, np.sqrt(rCGM**2 - b**2)])

    cloud_intersect = []
    for num, cloud in enumerate(clouds):
        clpos_x = cloud["x"]
        clpos_y = cloud["y"]
        clpos_z = cloud["z"]
        point = np.array([clpos_x, clpos_y, clpos_z])

        distance = np.linalg.norm(np.cross((point - r1), (r2 - r1))) / np.linalg.norm(
            r2 - r1
        )
        if distance < cloud_size:
            nIon_cloud = cloud["nIon"]
            cloud_intersect.append(
                [num, 2 * np.sqrt(cloud_size**2 - distance**2), nIon_cloud]
            )
    return cloud_intersect


r0 = unmodified.Halo.r0 * unmodified.Halo.UNIT_LENGTH / kpc  # kpc
rCGM = 1.1 * unmodified.Halo.r200 * unmodified.Halo.UNIT_LENGTH / kpc  # kpc

fvh = 0.939
fvw = 0.060
fvc = 1 - (fvh + fvw)
n_wcl = int(1e3)

rwarm = rCGM * (fvc / n_wcl) ** (1.0 / 3)
print(f"Cloud size {rwarm*1e3:.1f} pc")

pos_warm: Union[list, np.ndarray] = [
    np.random.uniform(r0 + rwarm, rCGM - rwarm, n_wcl),
    np.random.uniform(0.0, np.pi, n_wcl),
    np.random.uniform(0.0, 2 * np.pi, n_wcl),
]
pos_warm = np.array(
    [
        pos_warm[0] * np.sin(pos_warm[1]) * np.cos(pos_warm[2]),
        pos_warm[0] * np.sin(pos_warm[1]) * np.sin(pos_warm[2]),
        pos_warm[0] * np.cos(pos_warm[1]),
    ]
).T

distance_cloud = np.sqrt(np.sum(pos_warm**2, axis=1))
met_cloud = metallicity(distance_cloud)
rho_cloud = density(distance_cloud) * (TmedVH / TmedVW)  # press-eq

nH_guess = np.array(
    [
        ((prs(distance_cloud[i]) / kB) / TmedVW)
        * (Xp(met_cloud[i]) * 0.61)
        / (mH / mp)  # guess
        for i in range(n_wcl)
    ]
)
# This is a local quantity for the cloud
nH_cloud = 10.0 ** np.array(
    [
        root(
            lambda LognH: (prs(distance_cloud[i]) / kB)
            * Xp(met_cloud[i])
            * mu(
                10.0**LognH,
                TmedVW,
                met_cloud[i],
                redshift,
                mode,
            )
            - 10.0**LognH * TmedVW * (mH / mp),
            np.log10(nH_guess[i]),
        ).x[0]
        for i in range(n_wcl)
    ]
)
nIon_cloud = np.array(
    [
        nIon(nH_cloud[i], TmedVW, met_cloud[i], redshift, mode, element=element)
        for i in range(n_wcl)
    ]
)
clouds = [
    {
        "x": pos_warm[i, 0],
        "y": pos_warm[i, 1],
        "z": pos_warm[i, 2],
        "dist": distance_cloud[i],
        "Temp": TmedVW,
        "prs": prs(distance_cloud[i]),
        "met": met_cloud[i],
        "rho": rho_cloud[i],
        "nH": nH_cloud[i],
        "nIon": nIon_cloud[i],
    }
    for i in range(n_wcl)
]
print("Cloud populating complete!")

phi = np.linspace(0, 2 * np.pi, 10)
impact = np.linspace(r0, rCGM - r0, 20)

col_dens = np.zeros((impact.shape[0], phi.shape[0]), dtype=np.float32)

for i in range(impact.shape[0]):
    for j in range(phi.shape[0]):
        los = {
            "b": impact[i],
            "phi": phi[j],
        }
        tot_proj_length = 2 * np.sqrt(rCGM**2 - impact[i] ** 2)
        cloud_intersect = intersect_clouds(clouds, rwarm, los, rCGM)
        if len(cloud_intersect) > 0:
            col_dens[i, j] = np.sum(
                np.array(cloud_intersect)[:, 1] * np.array(cloud_intersect)[:, 2]
            )
        else:
            col_dens[i, j] = 0.0  # nhot*tot_proj_length
col_dens *= kpc
# each row corresponds to same impact parameter

save_dic = {"impact": impact, "col_dens": col_dens, "rCGM": rCGM}

np.savez(
    f"./figures/randomSight_e.{element}.npz",
    impact=impact,
    col_dens=col_dens,
    rCGM=rCGM,
)

for i in range(impact.shape[0]):
    plt.scatter(
        (impact[i] / rCGM) * np.ones(col_dens.shape[1]),
        col_dens[i, :],
        color="tab:blue",
    )
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlabel("b/r200")
plt.ylabel(r"Column Density [$cm^{-2}$]")
plt.savefig(f"./figures/randomSight_e.{element}.png")
# plt.show()
