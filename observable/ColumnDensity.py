# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:48:58 2023

@author: alankar
"""

import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import root
from typing import Union, Optional, Any

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from astro_plasma import Ionization
from misc.constants import mp, mH, kpc, kB, Xp
from misc.template import modified_field


class ColumnDensity:
    _verbose = False

    def __init__(self: "ColumnDensity", redisProf: modified_field) -> None:
        self.redisProf = redisProf
        self.genProf: bool = True
        self.nHhot: Optional[np.ndarray] = None
        self.nHwarm: Optional[np.ndarray] = None
        self.field: Optional[str] = None
        self._setup_additional_fields: bool = True

    def _setup_profile(self: "ColumnDensity") -> None:
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift

        # Use low initial resolution for interpolation
        if self.genProf:
            rend = (
                1.05
                * self.redisProf.unmodified.rCGM
                * (self.redisProf.unmodified.UNIT_LENGTH / kpc)
            )
            if self._verbose:
                print("Doing one time profile calculation", flush=True)
            radius_ = np.logspace(np.log10(5.0), np.log10(rend), 20)  # kpc
            _ = self.redisProf.ProfileGen(radius_)
            self.genProf = False
            if self._verbose:
                print("Complete!", flush=True)

        self.nHhot_local = interp1d(
            self.redisProf.radius, self.redisProf.nHhot_local, fill_value="extrapolate"
        )
        self.nHwarm_local = interp1d(
            self.redisProf.radius, self.redisProf.nHwarm_local, fill_value="extrapolate"
        )
        self.prs_hot = interp1d(
            self.redisProf.radius, self.redisProf.prs_hot, fill_value="extrapolate"
        )
        self.prs_warm = interp1d(
            self.redisProf.radius, self.redisProf.prs_warm, fill_value="extrapolate"
        )
        self.Tcut = interp1d(
            self.redisProf.radius, self.redisProf.Tcut, fill_value="extrapolate"
        )
        self.metallicity = interp1d(
            self.redisProf.radius,
            self.redisProf.unmodified.metallicity,
            fill_value="extrapolate",
        )
        self.fvw = interp1d(
            self.redisProf.radius, self.redisProf.fvw, fill_value="extrapolate"
        )
        self.TmedVH = interp1d(
            self.redisProf.radius, self.redisProf.TmedVH, fill_value="extrapolate"
        )

        self.THotM = interp1d(
            self.redisProf.radius,
            (self.redisProf.prs_hot / (self.redisProf.nhot_local * kB)),
            fill_value="extrapolate",
        )
        self.Temp = self.redisProf.TempDist

        if not (hasattr(self, "radius")):
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
                if self._setup_additional_fields:
                    # This function must be called after the previous two commands
                    self._additional_fields(indx, r_val)
        else:
            if self._setup_additional_fields:
                for indx, r_val in enumerate(self.radius):
                    self._additional_fields(indx, r_val)
        if self._setup_additional_fields:
            self._interpolate_additional_fields()
            self._setup_additional_fields = False

    def _additional_fields(self: "ColumnDensity", indx: int, r_val: float):
        # Implemented by child classes
        pass

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
        # Convert to cm^-2
        if len(b_) == 1:
            return coldens[0] * kpc
        else:
            return coldens * kpc
