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
from itertools import product
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
prs0_by_kB = 30


# -----------------------------------------
TmedVH = pow(10.0, 5.754)
TmedVW = pow(10.0, 5.265)
TmedVC = pow(10.0, 4.101)
sig = 0.3

THotM = TmedVH * np.exp(-(sig**2) / 2)
TWarmM = TmedVW * np.exp(-(sig**2) / 2)
TColdM = TmedVC * np.exp(-(sig**2) / 2)

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

prs = interp1d(
    radius,
    prs0_by_kB * kB * np.ones_like(radius),  # prsTh,
    fill_value="extrapolate",
)  # CGS

metallicity = interp1d(
    radius,
    unmodified.metallicity,
    fill_value="extrapolate",
)

Z0 = 1.0
ZrCGM = 0.3
p = Z0 / ZrCGM
met_avg = (
    1.5
    * (p - (p**2 - 1) * np.arcsin(1.0 / np.sqrt(p**2 - 1)))
    * Z0
    * np.sqrt(p**2 - 1)
)

nH_guess = (prs0_by_kB / THotM) * (Xp(met_avg) * 0.61) / (mH / mp)  # guess

nH_hot = (
    10.0
    ** root(
        lambda LognH: prs0_by_kB
        * Xp(met_avg)
        * mu(
            10.0**LognH,
            THotM,
            met_avg,
            redshift,
            mode,
        )
        - 10.0**LognH * THotM * (mH / mp),
        np.log10(nH_guess),
    ).x[0]
)
mu_CGM = mu(nH_hot, THotM, met_avg, redshift, mode, part_type="all")

density = interp1d(
    radius,
    prs0_by_kB / THotM * mu_CGM * mp * np.ones_like(radius),
    fill_value="extrapolate",
)  # CGS

nIon_hot = nIon(nH_hot, THotM, met_avg, redshift, mode, element=element)

print("Hot phase calculation complete!")


def intersect_clouds(all_clouds, cloud_size, los, rCGM):
    b = los["b"]
    phi = los["phi"]

    los_x = b * np.cos(phi)
    los_y = b * np.sin(phi)

    r1 = np.array([los_x, los_y, 0.0])
    r2 = np.array([los_x, los_y, np.sqrt(rCGM**2 - b**2)])

    cloud_intersect = []
    for cloud_type, clouds in enumerate(all_clouds):
        for num, cloud in enumerate(clouds):
            point = np.array([cloud["x"], cloud["y"], cloud["z"]])

            distance = np.linalg.norm(
                np.cross((point - r1), (r2 - r1))
            ) / np.linalg.norm(r2 - r1)
            if distance < cloud_size[cloud_type]:
                nIon_cloud = cloud["nIon"]
                cloud_intersect.append(
                    [
                        cloud_type,
                        num,
                        2 * np.sqrt(cloud_size[cloud_type] ** 2 - distance**2),
                        nIon_cloud,
                        distance,
                        cloud_size[cloud_type],
                    ]
                )
        overlap = 0.0
        if len(cloud_intersect) > 1:
            for i, cloud_i in enumerate(cloud_intersect):
                cloud_type = cloud_i[0]
                num = cloud_i[1]
                cl1 = np.array(
                    [
                        all_clouds[cloud_type][num]["x"],
                        all_clouds[cloud_type][num]["y"],
                        all_clouds[cloud_type][num]["z"],
                    ]
                )
                d1 = cloud_i[4]
                rcl1 = cloud_i[5]
                for j, cloud_j in enumerate(cloud_intersect[i + 1 :]):
                    cloud_type = cloud_j[0]
                    num = cloud_j[1]
                    cl2 = np.array(
                        [
                            all_clouds[cloud_type][num]["x"],
                            all_clouds[cloud_type][num]["y"],
                            all_clouds[cloud_type][num]["z"],
                        ]
                    )
                    d2 = cloud_j[4]
                    rcl2 = cloud_j[5]
                    lhs = np.abs(((cl2 - cl1) @ (r2 - r1)) / np.linalg.norm(r2 - r1))
                    rhs = np.sqrt(rcl1**2 - d1**2) + np.sqrt(rcl2**2 - d2**2)
                    if lhs < rhs:
                        overlap += rhs - lhs
        # no_overlap = 0.
        # for cloud_type in range( len(n_cloud) ):
        #     no_overlap += (np.count_nonzero(np.array(cloud_intersect)[:,0]==cloud_type)*cloud_size[cloud_type])
        if len(cloud_intersect) > 0:
            no_overlap = np.sum(np.array(cloud_intersect)[:, 5])
        else:
            no_overlap = 0.0
        cloud_column = no_overlap - overlap
        total_column = 2 * np.linalg.norm(r2 - r1)
    return (cloud_intersect, cloud_column, total_column)


r0 = unmodified.Halo.r0 * unmodified.Halo.UNIT_LENGTH / kpc  # kpc
rCGM = 1.1 * unmodified.Halo.r200 * unmodified.Halo.UNIT_LENGTH / kpc  # kpc

fvh = 0.939
fvw = 0.060
fvc = 1 - (fvh + fvw)

# ----------- These are the free parameters -------------
n_warm = int(0)
n_cold = int(1e3)

r_warm = rCGM * (fvw / n_warm) ** (1.0 / 3) if n_warm > 0 else np.inf
r_cold = rCGM * (fvc / n_cold) ** (1.0 / 3) if n_cold > 0 else np.inf

if n_warm > 0:
    print(f"Warm cloud size {r_warm*1e3:.1f} pc")
if n_cold > 0:
    print(f"Cold cloud size {r_cold*1e3:.1f} pc")


def _populate_clouds(n_clouds, r_cloud):
    pos_cloud: Union[list, np.ndarray] = [
        np.random.uniform(r0 + r_cloud, rCGM - r_cloud, n_clouds),
        np.random.uniform(0.0, np.pi, n_clouds),
        np.random.uniform(0.0, 2 * np.pi, n_clouds),
    ]
    pos_cloud = np.array(
        [
            pos_cloud[0] * np.sin(pos_cloud[1]) * np.cos(pos_cloud[2]),
            pos_cloud[0] * np.sin(pos_cloud[1]) * np.sin(pos_cloud[2]),
            pos_cloud[0] * np.cos(pos_cloud[1]),
        ]
    ).T
    return pos_cloud


n_cloud = []
r_cloud = []
pos_cloud = []
T_cloud = []
if n_warm > 0:
    pos_warm = _populate_clouds(n_warm, r_warm)
    pos_cloud.append(pos_warm)
    n_cloud.append(n_warm)
    r_cloud.append(r_warm)
    T_cloud.append(TWarmM)
if n_cold > 0:
    pos_cold = _populate_clouds(n_cold, r_cold)
    pos_cloud.append(pos_cold)
    n_cloud.append(n_cold)
    r_cloud.append(r_cold)
    T_cloud.append(TColdM)

distance_cloud = [np.sqrt(np.sum(pos**2, axis=1)) for pos in pos_cloud]

met_cloud = [metallicity(distance) for distance in distance_cloud]

rho_cloud = [
    density(distance) * (THotM / T_cloud[cloud_type])  # press-eq
    for cloud_type, distance in enumerate(distance_cloud)
]

nH_guess = [
    np.array(
        [
            ((prs(distance[i]) / kB) / T_cloud[cloud_type])
            * (Xp(met_cloud[cloud_type][i]) * 0.61)
            / (mH / mp)  # guess
            for i in range(n_cloud[cloud_type])
        ]
    )
    for cloud_type, distance in enumerate(distance_cloud)
]

# This is a local quantity for the cloud
nH_cloud = [
    10.0
    ** np.array(
        [
            root(
                lambda LognH: (prs(distance[i]) / kB)
                * Xp(met_cloud[cloud_type][i])
                * mu(
                    10.0**LognH,
                    T_cloud[cloud_type],
                    met_cloud[cloud_type][i],
                    redshift,
                    mode,
                )
                - 10.0**LognH * T_cloud[cloud_type] * (mH / mp),
                np.log10(nH_guess[cloud_type][i]),
            ).x[0]
            for i in range(n_cloud[cloud_type])
        ]
    )
    for cloud_type, distance in enumerate(distance_cloud)
]
nIon_cloud = [
    np.array(
        [
            nIon(
                nH_cloud[cloud_type][i],
                T_cloud[cloud_type],
                met_cloud[cloud_type][i],
                redshift,
                mode,
                element=element,
            )
            for i in range(n_cloud[cloud_type])
        ]
    )
    for cloud_type, distance in enumerate(distance_cloud)
]


clouds = [
    [
        {
            "x": pos_cloud[cloud_type][i, 0],
            "y": pos_cloud[cloud_type][i, 1],
            "z": pos_cloud[cloud_type][i, 2],
            "dist": distance[i],
            "Temp": T_cloud[cloud_type],
            "prs": prs(distance[i]),
            "met": met_cloud[cloud_type][i],
            "rho": rho_cloud[cloud_type][i],
            "nH": nH_cloud[cloud_type][i],
            "nIon": nIon_cloud[cloud_type][i],
        }
        for i in range(n_cloud[cloud_type])
    ]
    for cloud_type, distance in enumerate(distance_cloud)
]
print("Cloud populating complete!")

phi = np.linspace(0, 2 * np.pi, 200)
impact = np.linspace(r0, rCGM - r0, 100)
intersect_count = np.zeros((len(n_cloud), impact.shape[0], phi.shape[0]))


def generate_column(tup):
    i, j = tup
    los = {
        "b": impact[i],
        "phi": phi[j],
    }
    cloud_intersect, cloud_column, total_column = intersect_clouds(
        clouds, r_cloud, los, rCGM
    )
    if len(cloud_intersect) > 0:
        for cloud_type in range(len(n_cloud)):
            intersect_count[cloud_type, i, j] = np.count_nonzero(
                np.array(cloud_intersect)[:, 0] == cloud_type
            )
    column = 0.0
    if len(cloud_intersect) > 0:
        cloud_intersect = np.array(cloud_intersect)[:, 2:4]
        column += np.sum(np.product(cloud_intersect, axis=1))
    column += (total_column - cloud_column) * nIon_hot
    return column


col_dens = np.array(
    list(map(generate_column, product(range(impact.shape[0]), range(phi.shape[0]))))
).reshape((impact.shape[0], phi.shape[0]))
col_dens *= kpc
# each row corresponds to same impact parameter (impact.shape[0], phi.shape[0])

save_dic = {"impact": impact, "col_dens": col_dens, "rCGM": rCGM, "ncl": n_cold}

np.savez(
    f"./figures/randomSight_e.{element}_{n_cold}.npz",
    impact=impact,
    col_dens=col_dens,
    rCGM=rCGM,
    ncl=n_cold,
)

med = np.zeros_like(impact)
avg = np.zeros_like(impact)
for i in range(impact.shape[0]):
    n_intersect = intersect_count[:, i, :]
    non_zero_col = col_dens[i, :][np.sum(n_intersect, axis=0) > 0]
    if len(non_zero_col) == 0:
        continue
    plt.scatter(
        (impact[i] / rCGM) * np.ones(non_zero_col.shape[0]),
        non_zero_col,
        color="tab:blue",
    )
    med[i] = np.median(non_zero_col)  # col_dens[i, :]) #
    avg[i] = np.average(non_zero_col)  # col_dens[i, :])

plt.plot(impact / rCGM, med, color="tab:red", linestyle=":", linewidth=3)
plt.plot(impact / rCGM, avg, color="tab:red", linestyle="-", linewidth=3)

plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlabel("b/r200")
plt.ylabel(r"Column Density [$cm^{-2}$]")
plt.savefig(f"./figures/randomSight_e.{element}_w{n_warm}c{n_cold}.png")
# plt.show()
