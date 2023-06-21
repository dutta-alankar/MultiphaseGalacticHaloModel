# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:56:31 2022

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
from misc.constants import kpc
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from modified.isobarcool import IsobarCoolRedistribution
from observable.sb import SB_gen
from misc.template import unmodified_field, modified_field


def spectrum(unmod: str, mod: str, ionization: str) -> None:
    print(unmod, mod, ionization)

    cutoff = 4.0
    TmedVW = 3.0e5
    sig = 0.3
    redshift = 0.0

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

    modified: modified_field = None
    if mod == "isochor":
        modified = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    else:
        modified = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff)

    spectrum = SB_gen(modified)
    
    with open(f"figures/SB_{unmod}_{mod}_{ionization}.pickle", "wb") as f:
        data = {
            "energy": spectrum[:, 0],
            "sb_hot": spectrum[:, 1],
            "sb_warm": spectrum[:, 2],
            "rCGM": unmodified.rCGM * unmodified.UNIT_LENGTH / kpc,
        }
        pickle.dump(data, f)
    

if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor", "isobar"]
    ionization = ["PIE", "CIE"]

    for condition in product(unmod, mod, ionization):
        spectrum(*condition)
        
