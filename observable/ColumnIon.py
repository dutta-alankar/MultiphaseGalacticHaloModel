# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:53:50 2022

@author: Alankar
"""

import sys
import numpy as np
import os
from scipy.interpolate import interp1d
from typing import Union, Callable, Optional, Any

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
        self.nIon: Optional[Union[np.ndarray, Callable]] = None

    def _reset(self: "ion_column") -> None:
        super()._reset()
        self.file_path = None
        self.element = None
        self.ion = None
        self.field = "nIon"
        self.nIon = None

    def _additional_fields(self: "ion_column", indx: int, r_val: float) -> None:
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        nIon = Ionization.interpolate_num_dens

        if self.element is None:
            raise ValueError("Error: element not set!")

        if not (isinstance(self.nIon, np.ndarray)):
            self.nIon = np.zeros_like(self.radius)

        nIonWarm = np.array(
            [
                nIon(
                    self.nHwarm[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    mode=mode,
                    element=self.element,
                    ion=self.ion,
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        nIonHot = np.array(
            [
                nIon(
                    self.nHhot[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    mode=mode,
                    element=self.element,
                    ion=self.ion,
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        _, gvh, gvw = self.redisProf.probability_ditrib_mod(
            r_val,
            ThotM=self.ThotM,
            fvw=self.fvw,
            Temp=self.Temp,
            xmin=self.xmin,
            Tcutoff=self.Tcut,
        )
        TmedVH = self.ThotM(r_val) * np.exp(self.redisProf.sigH**2 / 2)
        xh = np.log(self.Temp / TmedVH)
        xw = np.log(self.Temp / self.redisProf.TmedVW)

        # Global density sensitive. The extra volume fraction factor is due to that
        hotInt = (1 - self.fvw(r_val)) * np.trapz(nIonHot * gvh, xh)
        warmInt = self.fvw(r_val) * np.trapz(nIonWarm * gvw, xw)

        self.nIon[indx] = hotInt + warmInt

    def _interpolate_additional_fields(self: "ion_column") -> None:
        self.nIon = interp1d(self.radius, self.nIon, fill_value="extrapolate")

    def gen_column(
        self: "ion_column",
        b_: Union[list, np.ndarray, float, int],
        *args: Any,
        **kwargs: Any
    ) -> Union[np.ndarray, float]:
        if self.field is None:
            raise AttributeError(
                "Error: Column calculating field not set!"
            )  # Set in __init__
        element: Union[int, str, AtmElement] = kwargs["element"]
        if "ion" in kwargs.keys():
            ion: Optional[int] = kwargs["ion"]
        else:
            ion = None
        element, ion = parse_atomic_ion_no(element, ion)
        self.element = element
        self.ion = ion
        self.nIon = None
        return super().gen_column(
            b_,
        )
