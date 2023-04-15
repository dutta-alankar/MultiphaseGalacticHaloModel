# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:53:50 2022

@author: Alankar
"""

import sys
import numpy as np
import os
from scipy.interpolate import interp1d
from typing import Union, Optional, Any

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from pathlib import Path
from astro_plasma import Ionization
from misc.constants import mp, mH, kB, Xp
from misc.template import modified_field
from observable.ColumnDensity import ColumnDensity
from astro_plasma.core.utils import AtmElement, parse_atomic_ion_no


class ion_column(ColumnDensity):
    def __init__(self: "ion_column", redisProf: modified_field):
        super().__init__(redisProf)
        self.file_path: Optional[Union[str, Path]] = None
        self.element: Optional[int] = None
        self.ion: Optional[int] = None
        self.field: str = "nIon"

    def _additional_fields(self: "ion_column", indx: int, r_val: float) -> None:
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        fIon = Ionization.interpolate_ion_frac
        mu = Ionization.interpolate_mu

        if self.file_path is None:
            self.file_path = os.path.realpath(__file__)
            dir_loc = os.path.split(self.file_path)[:-1]
            abn_file = os.path.join(
                *dir_loc,
                "..",
                "submodules",
                "AstroPlasma",
                "astro_plasma",
                "data",
                "solar_GASS10.abn"
            )
            _tmp = None
            with open(abn_file, "r") as file:
                _tmp = file.readlines()
            self.abn = np.array(
                [float(element.split()[-1]) for element in _tmp[2:32]]
            )  # Till Zinc

        a0 = self.abn[self.element - 1]

        if not (hasattr(self, "fIonHot")):
            self.fIonHot = np.zeros_like(self.nHhot)
        if not (hasattr(self, "mu_hot")):
            self.mu_hot = np.zeros_like(self.nHhot)
        if not (hasattr(self, "fIonWarm")):
            self.fIonWarm = np.zeros_like(self.nHwarm)
        if not (hasattr(self, "mu_warm")):
            self.mu_warm = np.zeros_like(self.nHwarm)
        if not (hasattr(self, "nIon")):
            self.nIon = np.zeros_like(self.radius)

        xh = np.log(
            self.Temp / (self.THotM(r_val) * np.exp(self.redisProf.sigH**2 / 2))
        )
        PvhT = np.exp(-(xh**2) / (2 * self.redisProf.sigH**2)) / (
            self.redisProf.sigH * np.sqrt(2 * np.pi)
        )
        xw = np.log(self.Temp / self.redisProf.TmedVW)
        gvwT = (
            self.fvw(r_val)
            * np.exp(-(xw**2) / (2 * self.redisProf.sigW**2))
            / (self.redisProf.sigW * np.sqrt(2 * np.pi))
        )
        gvhT = np.piecewise(
            PvhT,
            [
                self.Temp >= self.Tcut(r_val),
            ],
            [lambda xp: xp, lambda xp: 0.0],
        )

        self.fIonHot[indx, :] = 10.0 ** np.array(
            [
                fIon(
                    self.nHhot[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    self.element,
                    self.ion,
                    mode,
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        self.mu_hot[indx, :] = np.array(
            [
                mu(
                    self.nHhot[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    mode,
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        self.fIonWarm[indx, :] = 10.0 ** np.array(
            [
                fIon(
                    self.nHwarm[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    self.element,
                    self.ion,
                    mode,
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        self.mu_warm[indx, :] = np.array(
            [
                mu(
                    self.nHwarm[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    mode,
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        hotInt = np.trapz(
            (self.mu_hot[indx, :] * self.prs_hot(r_val) / (kB * self.Temp))
            * self.fIonHot[indx, :]
            * gvhT,
            xh,
        )  # Global density sensitive (1-self.fvw(r_val))*
        warmInt = np.trapz(
            (self.mu_warm[indx, :] * self.prs_warm(r_val) / (kB * self.Temp))
            * self.fIonWarm[indx, :]
            * gvwT,
            xw,
        )  # self.fvw(r_val)*

        self.nIon[indx] = (
            a0
            * self.metallicity(r_val)
            * (hotInt + warmInt)
            * (mp / mH)
            * Xp(self.metallicity(r_val))
        )

    def _interpolate_additional_fields(self: "ion_column") -> None:
        self.nIon = interp1d(self.radius, self.nIon, fill_value="extrapolate")

    def gen_column(
        self: "ion_column",
        b_: Union[list, np.ndarray, float, int],
        *args: Any,
        **kwargs: Any
    ) -> Union[np.ndarray, float]:
        if self.field is None:
            raise AttributeError("Error: Column calculating field not set!")
        element: Union[int, str, AtmElement] = kwargs["element"]
        if "ion" in kwargs.keys():
            ion: Optional[int] = kwargs["ion"]
        else:
            ion = None
        element, ion = parse_atomic_ion_no(element, ion)
        if self.element is not None and self.ion is not None:
            if self.element == element and self.ion == ion:
                self._setup_additional_fields = False
        else:
            self.element = element
            self.ion = ion
            self._setup_additional_fields = True
        return super().gen_column(
            b_,
        )
