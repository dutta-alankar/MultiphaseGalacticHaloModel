# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:15:09 2022

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
class IsochorCoolRedistribution(Redistribution):
    def __post_init__(self: "IsochorCoolRedistribution") -> None:
        super().__post_init__()
        self.isobaric: int = 0

    def _required_additional_props(self: "IsochorCoolRedistribution") -> None:
        self.redisType: str = "isochor"

    def _calculate_filling_fraction(
        self: "IsochorCoolRedistribution", xmin: float, indx: int
    ) -> Tuple[float, float]:
        fmw = 0.5 * (1 + erf((xmin + self.sig**2) / (np.sqrt(2) * self.sig)))
        fvw = 0.5 * (1 + erf(xmin / (np.sqrt(2) * self.sig)))
        return (fmw, fvw)
