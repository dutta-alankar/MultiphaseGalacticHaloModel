#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 10:22:13 2022

@author: alankar
"""

import sys
sys.path.append('../..')
import numpy as np
from typing import Union, List, Callable
from observable.measures import Measure

class DispersionMeasure(Measure):
    
    def observable_quantity(self: "DispersionMeasure", 
                            funcs: List[Callable], 
                            distance: Union[float, list, np.ndarray],
                            LOS_sample: Union[list, np.ndarray],) -> float:
        ne_prof = funcs[0]
        ne = np.nan_to_num(10.**ne_prof(np.log10(distance)))
        return np.trapz( ne, LOS_sample) # cm^-3 kpc
    
    def post_process_observable(self: "DispersionMeasure", 
                                quantity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (quantity * 1e3) # convert to cm^-3 pc
        