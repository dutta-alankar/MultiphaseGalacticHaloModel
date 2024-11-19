# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:15:58 2023

@author: alankar
"""
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Union
from misc.constants import kpc, kB
from misc.template import modified_field

def _get_lim(y: np.ndarray, extrapolate: bool = True) -> Union[tuple, str]:
    if extrapolate: 
        return "extrapolate"
    else:
        return (y[0], y[-1])

class _interpolate_internal_variables:
    _verbose: bool = True
    genProf: bool = True

    def __init__(self):
        self.redisProf: Optional[modified_field] = None

    def _interpolate_vars(self: "_interpolate_internal_variables") -> float:
        rend = (
            1.1
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

        _quan = self.redisProf.nHhot_local
        self.nHhot_local = interp1d(
            self.redisProf.radius, self.redisProf.nHhot_local, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.nHwarm_local
        self.nHwarm_local = interp1d(
            self.redisProf.radius, self.redisProf.nHwarm_local, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.prs_hot
        self.prs_hot = interp1d(
            self.redisProf.radius, self.redisProf.prs_hot, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.prs_warm
        self.prs_warm = interp1d(
            self.redisProf.radius, self.redisProf.prs_warm, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.Tcut
        self.Tcut = interp1d(
            self.redisProf.radius, self.redisProf.Tcut, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.xmin
        self.xmin = interp1d(
            self.redisProf.radius, self.redisProf.xmin, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.unmodified.metallicity
        self.metallicity = interp1d(
            self.redisProf.radius,
            self.redisProf.unmodified.metallicity,
            bounds_error=False, 
            fill_value=_get_lim(_quan),
        )

        _quan = self.redisProf.fvw
        self.fvw = interp1d(
            self.redisProf.radius, self.redisProf.fvw, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.TmedVH
        self.TmedVH = interp1d(
            self.redisProf.radius, self.redisProf.TmedVH, 
            bounds_error=False, fill_value=_get_lim(_quan)
        )

        _quan = self.redisProf.prs_hot / (self.redisProf.nhot_local * kB)
        self.ThotM = interp1d(
            self.redisProf.radius,
            (self.redisProf.prs_hot / (self.redisProf.nhot_local * kB)),
            bounds_error=False, 
            fill_value=_get_lim(_quan),
        )
        self.Temp = self.redisProf.TempDist

        return rend
