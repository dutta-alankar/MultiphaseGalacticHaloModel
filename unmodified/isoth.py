# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:39:12 2022

@author: Alankar
"""

import numpy as np
import sys

sys.path.append("..")
from misc.constants import mp, mH, kpc, K, kB, Xp
from scipy.optimize import root
from typing import Union, Tuple, Optional
from dataclasses import dataclass

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from astro_plasma import Ionization
from unmodified.unmodified_profile import UnmodifiedProfile


@dataclass
class IsothermalUnmodified(UnmodifiedProfile):
    # Thot is mass wighted average temperature in FM17 (not median Temperature!)
    # P0Tot is in units of kB K cm^-3
    THot: float = 1.43e6
    P0Tot: float = 4580.0
    alpha: float = 1.9

    def __post_init__(self: "IsothermalUnmodified") -> None:
        super().__post_init__()
        self.THot: float = self.THot * K
        self.P0Tot: float = (
            self.P0Tot * kB / (self.UNIT_DENSITY * self.UNIT_VELOCITY**2)
        )
        # self.sigmaTh = 141*(km/s)/self.__class__.UNIT_VELOCITY # FSM17 used this adhoc
        # mu here is assumed constant to have sigmaTh a radius independent quantity, hence unmodified profiles are analytic
        mu = 0.61
        self.sigmaTh: float = np.sqrt(kB * self.THot / (mu * mp)) / self.UNIT_VELOCITY
        self.unmod_type: str = "isoth"

        self.rho: Optional[np.ndarray] = None
        self.prsTh: Optional[np.ndarray] = None
        self.prsnTh: Optional[np.ndarray] = None
        self.prsTurb: Optional[np.ndarray] = None
        self.prsTot: Optional[np.ndarray] = None
        self.nH: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None

    def ProfileGen(
        self: "IsothermalUnmodified", radius_: Union[float, int, list, np.ndarray]
    ) -> Tuple:  # Takes in r in kpc, returns Halo density and pressure  in CGS
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

        prsTot = (
            self.P0Tot
            * np.exp(
                (
                    (
                        self.Halo.Phi(self.radius)
                        - self.Halo.Phi(self.Halo.r0 * self.Halo.UNIT_LENGTH / kpc)
                    )
                    / self.__class__.UNIT_VELOCITY**2
                )
                / (self.alpha * self.sigmaTh**2 + self.sigmaTurb**2)
            )
            * (self.__class__.UNIT_DENSITY * self.__class__.UNIT_VELOCITY**2)
        )  # CGS
        prsTh = prsTot / (self.alpha + (self.sigmaTurb / self.sigmaTh) ** 2)
        prsnTh = (self.alpha - 1) * prsTh
        prsTurb = prsTot - (prsTh + prsnTh)
        self.metallicity = self.Z0 / np.sqrt(1 + (self.radius / self.rZ) ** 2)

        ndens = prsTh / (kB * self.THot)
        nH_guess = (
            ndens * Xp(self.metallicity) * (mp / mH) * 0.61
        )  # guess nH with a guess mu
        nH = np.zeros_like(nH_guess)
        mu = Ionization.interpolate_mu
        for i in range(self.radius.shape[0]):

            def transcendental(LognH):
                return (
                    LognH
                    - np.log10(ndens[i])
                    - np.log10(
                        Xp(self.metallicity[i])
                        * (mp / mH)
                        * mu(
                            10.0**LognH,
                            self.THot,
                            self.metallicity[i],
                            self.redshift,
                            self.ionization,
                        )
                    )
                )

            nH[i] = 10.0 ** np.array([root(transcendental, np.log10(nH_guess[i])).x[0]])

        mu = np.array(
            [
                mu(
                    nH[i],
                    self.THot,
                    self.metallicity[i],
                    self.redshift,
                    self.ionization,
                )
                for i in range(self.radius.shape[0])
            ]
        )
        rho = ndens * mu * mp  # Gas density only includes thermal component
        self.Temperature = self.THot * np.ones_like(rho)
        self.rho = rho
        self.prsTh = prsTh
        self.prsnTh = prsnTh
        self.prsTurb = prsTurb
        self.prsTot = prsTot
        self.nH = nH
        self.mu = mu
        self.ndens = ndens

        return (rho, prsTh, prsnTh, prsTurb, prsTot, nH, mu)
