# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:18:30 2022

@author: Alankar
"""

import numpy as np
import sys

sys.path.append("..")
from misc.constants import mp, mH, kpc, K, cm, kB, Xp
from scipy.optimize import root
from dataclasses import dataclass
from typing import Union, Optional

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from astro_plasma import Ionization
from unmodified.unmodified_profile import UnmodifiedProfile

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
class IsentropicUnmodified(UnmodifiedProfile):
    _type: str = "isent"
    nHrCGM: float = 1.1e-5
    TthrCGM: float = 2.4e5

    def __post_init__(self: "IsentropicUnmodified") -> None:
        super().__post_init__()
        self.fZ: float = 0.012
        self.alpharCGM: float = 2.1
        self.gammaTh: float = 5 / 3
        self.gammaNTh: float = 4 / 3
        self.TthrCGM: float = (
            self.TthrCGM * K
        )  # 3.4e5*K*(mu/0.59)*(M200*UNIT_MASS/(1e12*MSun))*(rCGM*UNIT_LENGTH/(300*kpc))**(-1)
        self.nHrCGM: float = self.nHrCGM * (cm ** (-3)) / (self.UNIT_LENGTH ** (-3))

        mu = Ionization.interpolate_mu
        self.metallicityrCGM: float = self.Z0 / np.sqrt(1 + (self.rCGM / self.rZ) ** 2)
        self.murCGM: float = mu(
            self.nHrCGM * (self.UNIT_LENGTH ** (-3)),
            self.TthrCGM,
            self.metallicityrCGM,
            self.redshift,
            self.ionization,
        )
        self.nrCGM: float = (
            self.nHrCGM * (mH / mp) / (Xp(self.metallicityrCGM) * self.murCGM)
        )  # code
        self.rhorCGM: float = (
            self.nHrCGM
            * mH
            / Xp(self.metallicityrCGM)
            * (self.UNIT_LENGTH ** (-3))
            / self.UNIT_DENSITY
        )
        self.KTh: float = (
            kB * self.TthrCGM / (self.UNIT_MASS * self.UNIT_VELOCITY**2)
        ) / (
            ((self.murCGM * mp / self.UNIT_MASS) ** self.gammaTh)
            * (self.nrCGM ** (self.gammaTh - 1))
        )  # code
        self.KNTh: float = (
            (kB * self.TthrCGM / (self.UNIT_MASS * self.UNIT_VELOCITY**2))
            / (
                ((self.murCGM * mp / self.UNIT_MASS) ** self.gammaNTh)
                * (self.nrCGM ** (self.gammaNTh - 1))
            )
        ) * (
            self.alpharCGM - 1
        )  # code
        self.Db: float = (
            self.sigmaTurb**2 * np.log(self.rhorCGM)
            + self.KTh
            * ((self.gammaTh) / (self.gammaTh - 1))
            * (self.rhorCGM ** (self.gammaTh - 1))
            + self.KNTh
            * ((self.gammaNTh) / (self.gammaNTh - 1))
            * (self.rhorCGM ** (self.gammaNTh - 1))
        )  # code
        self.unmod_type: str = "isent"

        self.rho: Optional[np.ndarray] = None
        self.prsTh: Optional[np.ndarray] = None
        self.prsnTh: Optional[np.ndarray] = None
        self.prsTurb: Optional[np.ndarray] = None
        self.prsTot: Optional[np.ndarray] = None
        self.nH: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.unmod_filename: str = f"unmod_{self._type}_ionization_{self.ionization}.pickle"

    def ProfileGen(
        self: "IsentropicUnmodified", radius_: Union[float, list, np.ndarray]
    ) -> tuple:
        # Takes in r in kpc, returns Halo density and pressure  in CGS
        super().ProfileGen(radius_)

        if (self._already_acomplished) and (self.rho is not None):
            return (
                self.rho,
                self.prsTh,
                self.prsnTh,
                self.prsTurb,
                self.prsTot,
                self.nH,
                self.mu,
            )

        mu = Ionization.interpolate_mu

        def _transcendental(log10rho: float, rad: float) -> float:
            rho = 10**log10rho
            return (
                self.sigmaTurb**2 * np.log(rho)
                + self.KTh
                * ((self.gammaTh) / (self.gammaTh - 1))
                * (rho ** (self.gammaTh - 1))
                + self.KNTh
                * ((self.gammaNTh) / (self.gammaNTh - 1))
                * (rho ** (self.gammaNTh - 1))
                - self.Db
                + (
                    (
                        -self.Halo.Phi(rad * self.UNIT_LENGTH / kpc)
                        + self.Halo.Phi(self.rCGM * self.UNIT_LENGTH / kpc)
                    )
                    / self.UNIT_VELOCITY**2
                )
            )

        nH_guess = np.logspace(
            -3 * (8.5 / (self.radius[0] * self.UNIT_LENGTH / kpc)),
            -5 * (220 / (self.radius[-1] * self.UNIT_LENGTH / kpc)),
            len(self.radius),
        )  # CGS
        rho_guess = np.log10((nH_guess * mH / Xp(self.metallicity)) / self.UNIT_DENSITY)

        rho = np.zeros_like(rho_guess)
        for indx in range(rank, self.radius.shape[0], size):
            rad = self.radius[indx]
            rho[indx] = 10.0 ** (root(_transcendental, rho_guess[indx], args=(rad,)).x[0]) # code
        if _mpi:
            comm.Barrier()
            # use MPI to get the totals
            _tmp = np.zeros_like(rho)
            comm.Allreduce([rho, MPI.DOUBLE], [_tmp, MPI.DOUBLE], op=MPI.SUM)
            rho = np.copy(_tmp)

        self.prsTh = (
            self.KTh
            * (rho**self.gammaTh)
            * self.UNIT_DENSITY
            * self.UNIT_VELOCITY**2
        )
        self.prsnTh = (
            self.KNTh
            * (rho**self.gammaNTh)
            * self.UNIT_DENSITY
            * self.UNIT_VELOCITY**2
        )
        self.prsTurb = (
            rho * self.sigmaTurb**2 * self.UNIT_DENSITY * self.UNIT_VELOCITY**2
        )
        self.prsTot = self.prsTh + self.prsnTh + self.prsTurb

        self.rho = rho * self.UNIT_DENSITY
        self.nH = self.rho * Xp(self.metallicity) / mH

        self.Temperature = np.zeros_like(self.radius)


        for i in range(rank, self.radius.shape[0], size):

            def transcendental(LogTemp):
                return np.log10(
                    10.0**LogTemp
                    / mu(
                        self.nH[i],
                        10.0**LogTemp,
                        self.metallicity[i],
                        self.redshift,
                        self.ionization,
                    )
                ) - np.log10(self.prsTh[i] / kB / (self.rho[i] / mp))

            logT_guess = np.log10(((self.prsTh[i] / kB) / (self.rho[i] / mp)) * 0.61)
            self.Temperature[i] = 10.0 ** (
                root(transcendental, logT_guess, method="lm", tol=1e-8).x[0]
            )

        if _mpi:
            comm.Barrier()
            # use MPI to get the totals
            _tmp = np.zeros_like(self.radius)
            comm.Allreduce([self.Temperature, MPI.DOUBLE], [_tmp, MPI.DOUBLE], op=MPI.SUM)
            self.Temperature = np.copy(_tmp)
        self.mu = self.Temperature / ((self.prsTh / kB) / (self.rho / mp))
        self.ndens = self.rho / (self.mu * mp)

        return (
            self.rho,
            self.prsTh,
            self.prsnTh,
            self.prsTurb,
            self.prsTot,
            self.nH,
            self.mu,
        )
