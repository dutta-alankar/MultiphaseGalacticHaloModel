# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:12:27 2023

@author: alankar
"""
import numpy as np
from typing import Protocol, Optional, Union, Any, Tuple


class dm_halo(Protocol):
    r0: float
    r200: float
    UNIT_LENGTH: float

    def Phi(self: "dm_halo", r: Union[float, int, list, np.ndarray]) -> np.ndarray:
        pass

    def Mass(
        self: "dm_halo",
        radius: Union[float, int, list[float], list[int], np.ndarray],
    ) -> Union[float, np.ndarray]:
        pass


class unmodified_field(Protocol):
    Halo: "dm_halo"
    rCGM: float
    ionization: str
    redshift: float
    metallicity: np.ndarray
    radius: np.ndarray
    unmod_type: str
    UNIT_LENGTH: float

    def ProfileGen(
        self: "unmodified_field",
        radius: Union[float, int, list[float], list[int], np.ndarray],
    ) -> Optional[tuple]:
        pass


class modified_field(Protocol):
    ionization: str
    redshift: float
    unmodified: "unmodified_field"
    UNIT_LENGTH: float
    TempDist: np.ndarray
    Tcut: np.ndarray
    prs_hot: np.ndarray
    prs_warm: np.ndarray
    nhot_local: np.ndarray
    sigH: float
    sigW: float
    TmedVW: float
    TmedVH: np.ndarray
    nHhot_local: np.ndarray
    nHwarm_local: np.ndarray
    fvw: np.ndarray
    radius: np.ndarray
    xmin: np.ndarray

    def ProfileGen(
        self: "modified_field",
        radius: Union[float, int, list[float], list[int], np.ndarray],
    ) -> Optional[tuple]:
        pass

    def probability_ditrib_mod(
        self: "modified_field",
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
