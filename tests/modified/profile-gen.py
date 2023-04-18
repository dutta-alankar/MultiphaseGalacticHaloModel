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
import pickle
from itertools import product
from typing import Optional
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isobarcool import IsobarCoolRedistribution
from modified.isochorcool import IsochorCoolRedistribution
from misc.template import unmodified_field, modified_field


def profile_gen(unmod: str, mod: str, ionization: str) -> None:
    print(unmod, mod, ionization)

    cutoff = 4.0
    TmedVW = 3.0e5
    sig = 0.3
    redshift = 0.2

    radius = np.linspace(9.0, 250, 30)  # kpc

    unmodified: Optional[unmodified_field] = None
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
    with open(f"figures/mod_prof_{unmod}_{mod}_{ionization}.pickle", "wb") as f:
        data = {
            "radius": radius,
            "nhot_local": nhot_local,
            "nwarm_local": nwarm_local,
            "nhot_global": nhot_global,
            "nwarm_global": nwarm_global,
        }
        pickle.dump(data, f)


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]

    for condition in product(unmod, mod, ionization):
        profile_gen(*condition)
