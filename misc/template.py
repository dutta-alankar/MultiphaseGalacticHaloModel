# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:12:27 2023

@author: alankar
"""
import numpy as np
from dataclasses import dataclass
from typing import Protocol, Optional, Union

@dataclass
class dm_halo(Protocol):
    r0: Optional[float] = None
    UNIT_LENGTH: Optional[float] = None
    
@dataclass
class unmodified_field(Protocol):
    Halo: Optional["dm_halo"] = None
    rCGM: Optional[float] = None
    metallicity: Optional[np.ndarray] = None
    UNIT_LENGTH: Optional[float] = None
    
    def ProfileGen(self, radius: Union[float, int, list[float], list[int], np.ndarray]):
        pass
    
@dataclass
class modified_field(Protocol):    
    ionization: Optional[float] = None
    redshift: Optional[float] = None
    unmodfied: Optional["unmodified_field"] = None
    UNIT_LENGTH: Optional[float] = None
    TempDist: Optional[np.ndarray] = None
    Tcut: Optional[np.ndarray] = None
    prs_hot: Optional[np.ndarray] = None
    prs_warm: Optional[np.ndarray] = None
    nhot_local: Optional[np.ndarray] = None
    sigH: Optional[float] = None
    sigW: Optional[float] = None
    TmedVW: Optional[float] = None
    TmedVH: Optional[float] = None
    nHhot_local: Optional[np.ndarray] = None
    nHwarm_local: Optional[np.ndarray] = None
    fvw: Optional[np.ndarray] = None
    
    def ProfileGen(self, radius: Union[float, int, list[float], list[int], np.ndarray]):
        pass