# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:15:58 2023

@author: alankar
"""
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional
from misc.constants import kpc, kB
from misc.template import modified_field


class _interpolate_internal_variables:
    _verbose: bool = True
    genProf: bool = True

    def __init__(self):
        self.redisProf: Optional[modified_field] = None

    def _interpolate_vars(self: "_interpolate_internal_variables") -> float:
        rend = (
            1.05
            * self.redisProf.unmodified.rCGM
            * (self.redisProf.unmodified.UNIT_LENGTH / kpc)
        )
        # Use low initial resolution for interpolation
        if self.genProf:
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
        self.xmin = interp1d(
            self.redisProf.radius, self.redisProf.xmin, fill_value="extrapolate"
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

        self.ThotM = interp1d(
            self.redisProf.radius,
            (self.redisProf.prs_hot / (self.redisProf.nhot_local * kB)),
            fill_value="extrapolate",
        )
        self.Temp = self.redisProf.TempDist

        return rend
