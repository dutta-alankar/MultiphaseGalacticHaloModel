# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:28:36 2022

@author: Alankar
"""

from scipy.special import erf
import numpy as np
from typing import Protocol, Optional, Tuple
from dataclasses import dataclass
from modified.redistribution import Redistribution

@dataclass
class unmodified_field(Protocol):
    redshift: Optional[float] = None
    ionization: Optional[float] = None

@dataclass
class IsobarCoolRedistribution(Redistribution):

    isobaric: int = 1
    
    def __post_init__(self: "IsobarCoolRedistribution") -> None:
        super().__post_init__() 
        
    def _required_additional_props(self: "IsobarCoolRedistribution") -> None:
        self.redisType: str = "isobar"
        
    def _calculate_filling_fraction(self: "IsobarCoolRedistribution", xmin: float, indx: int) -> Tuple[float,float]:
        fmw = 0.5*(1 + erf((xmin+self.sig**2)/(np.sqrt(2)*self.sig)))
        if (fmw>=0.5): fmw = 0.5
        fvw = (self.TmedVW/self.TmedVu[indx]) * np.exp(-(self.sigW**2 - self.sig**2)/2) * fmw
        return (fmw, fvw)
