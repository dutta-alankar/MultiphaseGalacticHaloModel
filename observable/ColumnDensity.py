# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:48:58 2023

@author: alankar
"""

import sys
import numpy as np
from scipy import integrate
from scipy.optimize import root
from typing import Union, Optional, Any
from abc import ABC, abstractmethod

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from astro_plasma import Ionization
from misc.constants import mp, mH, kpc, kB, Xp
from misc.template import modified_field
from observable.internal_interpolation import _interpolate_internal_variables


class ColumnDensity(ABC, _interpolate_internal_variables):
    _verbose: bool = True

    def __init__(self: "ColumnDensity", redisProf: modified_field) -> None:
        self.redisProf = redisProf
        self.nHhot: Optional[np.ndarray] = None
        self.nHwarm: Optional[np.ndarray] = None
        self.field: Optional[str] = None
        self.radius: Optional[np.ndarray] = None

    def _reset(self: "ColumnDensity") -> None:
        self.genProf = True
        self.nHhot = None
        self.nHwarm = None
        self.field = None
        self.radius = None

    def _setup_profile(self: "ColumnDensity") -> None:
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift

        rend = self._interpolate_vars()
        if self._verbose:
            print("Interpolation complete!")
        if self.radius is None:
            self.radius = np.logspace(np.log10(5.0), np.log10(rend), 80)  # kpc
        if self.nHhot is None or self.nHwarm is None:
            # Temp ...............
            # . R
            # . a
            # . d
            # .
            self.nHhot = np.zeros(
                (self.radius.shape[0], self.Temp.shape[0]), dtype=np.float64
            )
            self.nHwarm = np.zeros_like(self.nHhot)

            if self._verbose:
                print("Calculating nH hot/warm")
            mu = Ionization.interpolate_mu
            for indx, r_val in enumerate(self.radius):
                # Approximation is nH T is also constant like n T which is used as guess
                nHhot_guess = (
                    self.nHhot_local(r_val)
                    * self.TmedVH(r_val)
                    * np.exp(-self.redisProf.sigH**2 / 2)
                    / self.Temp
                )  # CGS
                nHwarm_guess = (
                    self.nHwarm_local(r_val)
                    * self.redisProf.TmedVW
                    * np.exp(-self.redisProf.sigW**2 / 2)
                    / self.Temp
                )  # CGS

                self.nHhot[indx, :] = 10.0 ** np.array(
                    [
                        root(
                            lambda LognH: (self.prs_hot(r_val) / (kB * self.Temp[i]))
                            * Xp(self.metallicity(r_val))
                            * mu(
                                10**LognH,
                                self.Temp[i],
                                self.metallicity(r_val),
                                redshift,
                                mode,
                            )
                            - (mH / mp) * (10**LognH),
                            np.log10(nHhot_guess[i]),
                        ).x[0]
                        for i in range(self.Temp.shape[0])
                    ]
                )
                self.nHwarm[indx, :] = 10.0 ** np.array(
                    [
                        root(
                            lambda LognH: (self.prs_warm(r_val) / (kB * self.Temp[i]))
                            * Xp(self.metallicity(r_val))
                            * mu(
                                10**LognH,
                                self.Temp[i],
                                self.metallicity(r_val),
                                redshift,
                                mode,
                            )
                            - (mH / mp) * (10**LognH),
                            np.log10(nHwarm_guess[i]),
                        ).x[0]
                        for i in range(self.Temp.shape[0])
                    ]
                )

        if self._verbose:
            print("Interpolating additional fields")
        for indx, r_val in enumerate(self.radius):
            # This function must be called after the previous two commands
            self._additional_fields(indx, r_val)
        self._interpolate_additional_fields()

    @abstractmethod
    def _additional_fields(self: "ColumnDensity", indx: int, r_val: float):
        # Implemented by child classes
        pass

    @abstractmethod
    def _interpolate_additional_fields(self: "ColumnDensity"):
        # Implemented by child classes
        pass

    def gen_column(
        self: "ColumnDensity",
        b_: Union[list, np.ndarray, float, int],
        *args: Any,
        **kwargs: Any,
    ) -> Union[np.ndarray, float]:
        # Takes in b_ in kpc, and the field name as string returns col dens of that field in CGS
        # `field` string name must match the `field` function created in `_setup_profile`
        if isinstance(b_, float) or isinstance(b_, int):
            b_ = np.array([b_])
        coldens = np.zeros_like(b_)
        self._setup_profile()
        if self._verbose:
            print("Profile setup complete!")

        field_func = eval(f"self.{self.field}")
        epsilon = 1e-6
        for indx, b_val in enumerate(b_):
            coldens[indx] = (
                2
                * integrate.quad(
                    lambda r: field_func(r) * r / np.sqrt(r**2 - b_val**2),
                    b_val * (1 + epsilon),
                    self.redisProf.unmodified.rCGM
                    * self.redisProf.unmodified.UNIT_LENGTH
                    / kpc,
                )[0]
            )  # kpc cm^-3
        if self._verbose:
            print("Column density calculation complete!")
        # Convert to cm^-2
        if len(b_) == 1:
            return coldens[0] * kpc
        else:
            return coldens * kpc
