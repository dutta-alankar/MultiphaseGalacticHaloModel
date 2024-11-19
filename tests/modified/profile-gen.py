# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:38:11 2023

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import numpy as np
import copy
import pickle
from itertools import product
from typing import Optional
from misc.constants import kpc
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isobarcool import IsobarCoolRedistribution
from modified.isochorcool import IsochorCoolRedistribution
from misc.template import unmodified_field, modified_field

_mpi = True

if _mpi:
    from mpi4py import MPI

    ## start parallel programming ---------------------------------------- #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    # t_start = MPI.Wtime()
else:
    rank = 0
    size = 1

Npoints = 5

def profile_gen(unmod: str, mod: str, ionization: str) -> None:
    if rank == 0:
        print(unmod, mod, ionization, flush=True)

    cutoff = 4.0
    TmedVW = 3.0e5
    sig = 0.3
    redshift = 0.0

    radius = np.linspace(9.0, 250, Npoints)  # kpc

    unmodified: Optional[unmodified_field] = None
    just_unmod: Optional[unmodified_field] = None
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

    just_unmod = copy.deepcopy(unmodified)
    modified: Optional[modified_field] = None
    if mod == "isochor":
        modified = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    else:
        modified = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff, isobaric=0)

    (
        nhot_local,
        nwarm_local,
        nhot_global,
        nwarm_global,
        fvw,
        fmw,
        prs_hot,
        prs_warm,
        _,
    ) = modified.ProfileGen(radius)
    modified.save()

    _ = just_unmod.ProfileGen(radius)
    n_unmod = just_unmod.ndens
    T_unmod = just_unmod.Temperature

    with open(f"figures/mod_prof_{unmod}_{mod}_{ionization}.pickle", "wb") as f:
        # print(unmodified.Halo.r200 * unmodified.Halo.UNIT_LENGTH / kpc, flush=True)
        data = {
            "radius": radius,
            "rvir": unmodified.Halo.r200 * unmodified.Halo.UNIT_LENGTH / kpc,
            "nhot_local": nhot_local,
            "nwarm_local": nwarm_local,
            "nhot_global": nhot_global,
            "nwarm_global": nwarm_global,
            "n_unmod": n_unmod,
            "T_unmod": T_unmod,
        }
        pickle.dump(data, f)


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]

    for condition in product(unmod, mod, ionization):
        profile_gen(*condition)
