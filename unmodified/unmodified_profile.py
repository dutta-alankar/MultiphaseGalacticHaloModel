# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:21:15 2023

@author: alankar
"""

import numpy as np
import sys
import pickle
import pathlib

sys.path.append("..")
from typing import Union, Optional
from dataclasses import dataclass
from misc.constants import mp, kpc, km, s, K
from misc.HaloModel import HaloModel
from misc.template import dm_halo

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

@dataclass
class UnmodifiedProfile:
    UNIT_LENGTH = kpc
    UNIT_DENSITY = mp
    UNIT_VELOCITY = km / s
    UNIT_MASS = UNIT_DENSITY * UNIT_LENGTH**3
    UNIT_TIME = UNIT_LENGTH / UNIT_VELOCITY
    UNIT_ENERGY = UNIT_MASS * (UNIT_LENGTH / UNIT_TIME) ** 2
    UNIT_TEMPERATURE = K

    sigmaTurb: float = 60.0
    Z0: float = 1.0
    ZrCGM: float = 0.3
    M200: float = 1e12
    MBH: float = 2.6e6
    rd: float = 3.0
    Mblg: float = 6e10
    r0: float = 8.5
    C: float = 12.0
    redshift: float = 0.0
    ionization: str = "PIE"
    _call_profile: bool = True

    def __post_init__(self: "UnmodifiedProfile") -> None:
        self.Halo: dm_halo = HaloModel(
            self.M200, self.MBH, self.Mblg, self.rd, self.r0, self.C
        )
        self.rCGM = 1.1 * self.Halo.r200 * self.Halo.UNIT_LENGTH / self.UNIT_LENGTH
        self.rZ = self.rCGM / np.sqrt((self.Z0 / self.ZrCGM) ** 2 - 1)
        self.sigmaTurb = self.sigmaTurb * (km / s) / self.UNIT_VELOCITY
        self.radius: Optional[np.ndarray] = None

    def save(self: "UnmodifiedProfile")-> None:
        self._call_profile = False
        if pathlib.Path(self.unmod_filename).is_file():
            return
        if rank == 0:
            # print(f"Saving {filename} ...", end=" ")
            with open(self.unmod_filename, "wb") as f:
                pickle.dump(self, f)
            # print("Done!")

    def ProfileGen(
        self: "UnmodifiedProfile", radius_: Union[float, int, list, np.ndarray]
    ) -> Optional[tuple]:
        # takes in r in kpc, returns Halo density and pressure  in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
        else:
            radius_ = np.array(radius_)

        # Check if already generated then don't redo any work
        self._already_acomplished = False
        if self.radius is not None:
            if type(self.radius) == list:
                self.radius = np.array(self.radius)
            if (
                np.sum(np.copy(radius_) * kpc / self.UNIT_LENGTH == self.radius)
                == self.radius.shape[0]
            ):
                self._already_acomplished = True

        self.radius = np.copy(radius_) * kpc / self.UNIT_LENGTH
        self.metallicity = self.Z0 / np.sqrt(
            1 + ((self.radius * kpc / self.UNIT_LENGTH) / self.rZ) ** 2
        )
        self._call_profile = False
        return None
