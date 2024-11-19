# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:34:18 2022

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import numpy as np
import pickle
from typing import Union, Optional
from itertools import product
from misc.constants import kpc
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from modified.isobarcool import IsobarCoolRedistribution
from observable.DispersionMeasure import DispersionMeasure as DM
from observable.EmissionMeasure import EmissionMeasure as EM
from observable.disk_measures import DiskDM, DiskEM
from misc.template import unmodified_field, modified_field

def gen_measure(
    unmod: str,
    mod: str,
    ionization: str,
    l: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
    map_type: str,
) -> None:
    print(map_type, unmod, mod, ionization)

    showProgress = False
    only_disk_update = False

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

    modified: Optional[modified_field] = None
    if mod == "isochor":
        modified = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    else:
        modified = IsobarCoolRedistribution(unmodified, TmedVW, sig, cutoff, isobaric=0)

    if not(only_disk_update):
        if map_type == "dispersion":
            map_val = DM(modified).make_map(l, b, showProgress=showProgress)
        else:
            map_val = EM(modified).make_map(l, b, showProgress=showProgress)
    else:
        with open(
        f"figures/map_{map_type}_{unmod}_{mod}_{ionization}.pickle", "rb"
        ) as data_file:
            data = pickle.load(data_file)
            map_val = data["map"]
  
    rvir = unmodified.Halo.r200 * (unmodified.Halo.UNIT_LENGTH / kpc)

    if map_type == "dispersion":
        disk = DiskDM(rvir=rvir).make_map(l, b, showProgress=showProgress)
    else:
        disk = DiskEM(rvir=rvir).make_map(l, b, showProgress=showProgress)

    with open(f"figures/map_{map_type}_{unmod}_{mod}_{ionization}.pickle", "wb"
              ) as data_file:
        data = {
            "l": l,
            "b": b,
            "map": map_val,
            "disk": disk,
        }
        # print(list(data.keys()))
        pickle.dump(data, data_file)


if __name__ == "__main__":
    unmod = ["isoth", "isent"]
    mod = ["isochor",]# "isobar"]
    ionization = ["PIE",]# "CIE"]

    b = np.linspace(-90, 90, 256)
    l = np.linspace(0, 360, 512)

    l, b = np.meshgrid(l, b)

    for condition in product(unmod, mod, ionization):
        gen_measure(*condition, l, b, "dispersion")
        gen_measure(*condition, l, b, "emission")
