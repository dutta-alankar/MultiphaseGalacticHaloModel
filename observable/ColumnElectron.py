# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:56:00 2022

@author: alankar
"""

import sys
import numpy as np
from typing import Optional, Any, Union, Callable
from scipy.interpolate import interp1d

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from astro_plasma import Ionization
from misc.template import modified_field
from observable.ColumnDensity import ColumnDensity


class electron_column(ColumnDensity):
    def __init__(self: "electron_column", redisProf: modified_field):
        super().__init__(redisProf)
        self.neHot: Optional[np.ndarray] = None
        self.neWarm: Optional[np.ndarray] = None
        self.ne: Optional[Union[np.ndarray, Callable]] = None
        self.field: str = "ne"

    def _reset(self: "electron_column"):
        super()._reset()
        self.neHot = None
        self.neWarm = None
        self.ne = None
        self.field = "ne"

    def _additional_fields(self: "electron_column", indx: int, r_val: float) -> None:
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        num_dens = Ionization.interpolate_num_dens

        if not (isinstance(self.ne, np.ndarray)):
            self.ne = np.zeros_like(self.radius)

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

        self.neHot = np.array(
            [
                num_dens(
                    self.nHhot[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    mode=mode,
                    part_type="electron",
                )
                for i in range(self.Temp.shape[0])
            ]
        )
        self.neWarm = np.array(
            [
                num_dens(
                    self.nHwarm[indx, i],
                    self.Temp[i],
                    self.metallicity(r_val),
                    redshift,
                    mode=mode,
                    part_type="electron",
                )
                for i in range(self.Temp.shape[0])
            ]
        )

        hotInt = (1 - self.fvw(r_val)) * np.trapz((self.neHot * gvhT, xh))
        # global density sensitive
        warmInt = self.fvw(r_val) * np.trapz((self.neWarm * gvwT, xw))
        self.ne[indx] = hotInt + warmInt

    def _interpolate_additional_fields(self: "electron_column"):
        self.ne = interp1d(self.radius, self.ne, fill_value="extrapolate")

    def gen_column(
        self: "electron_column",
        b_: Union[list, np.ndarray, float, int],
        *args: Any,
        **kwargs: Any
    ) -> Union[np.ndarray, float]:
        self.ne = None
        return super().gen_column(b_)
