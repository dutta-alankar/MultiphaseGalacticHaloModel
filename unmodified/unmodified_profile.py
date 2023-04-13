#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:21:15 2023

@author: alankar
"""

import numpy as np
import sys
sys.path.append('..')
from typing import Union
from dataclasses import dataclass
from misc.constants import mp, kpc, km, s, K
from misc.HaloModel import HaloModel

@dataclass
class UnmodifiedProfile:
    UNIT_LENGTH      = kpc
    UNIT_DENSITY     = mp
    UNIT_VELOCITY    = km/s
    UNIT_MASS        = UNIT_DENSITY*UNIT_LENGTH**3
    UNIT_TIME        = UNIT_LENGTH/UNIT_VELOCITY
    UNIT_ENERGY      = UNIT_MASS*(UNIT_LENGTH/UNIT_TIME)**2
    UNIT_TEMPERATURE = K
    
    sigmaTurb: float = 60.
    ZrCGM: float = 0.3
    M200: float = 1e12
    MBH: float = 2.6e6
    rd: float = 3.0
    Mblg: float = 6e10
    r0: float = 8.5
    C: float = 12.0
    redshift: float = 0.
    ionization: str = 'PIE'
    
    def __post_init__(self: "UnmodifiedProfile") -> None:
        self.Halo = HaloModel(self.M200, self.MBH, self.Mblg, self.rd, self.r0, self.C)
        self.rCGM = 1.1*self.Halo.r200*self.Halo.UNIT_LENGTH/self.UNIT_LENGTH
        self.Z0   = 1.0
        self.rZ    = self.rCGM/np.sqrt((self.Z0/self.ZrCGM)**2-1)
        self.sigmaTurb = self.sigmaTurb*(km/s)/self.UNIT_VELOCITY
        
    def ProfileGen(self: "UnmodifiedProfile", radius_: Union[float, list, np.ndarray]) -> None:  #takes in r in kpc, returns Halo density and pressure  in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
        
        # Check if already generated then don't redo any work
        self._alreadry_acomplished = False
        if hasattr(self, "radius"):
            if (np.sum(np.copy(radius_)*kpc/self.UNIT_LENGTH == self.radius) == self.radius.shape[0]):
                self._alreadry_acomplished = True
                
        self.radius = np.copy(radius_)*kpc/self.UNIT_LENGTH
        self.metallicity = self.Z0/np.sqrt(1+((self.radius*kpc/self.UNIT_LENGTH)/self.rZ)**2)