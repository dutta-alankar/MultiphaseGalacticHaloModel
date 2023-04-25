# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:22:42 2022

@author: alankar
"""

import sys

sys.path.append("..")
import numpy as np
from typing import List, Union, Callable
from observable.measures import Measure


class EmissionMeasure(Measure):
    def observable_quantity(
        self: "EmissionMeasure",
        funcs: List[Callable],
        distance: Union[float, list, np.ndarray],
        LOS_sample: Union[list, np.ndarray],
    ) -> float:
        neni_prof = funcs[1]
        neni = np.nan_to_num(10.0 ** neni_prof(np.log10(distance)))
        return np.trapz(neni, LOS_sample)  # cm^-6 kpc

    def post_process_observable(
        self: "EmissionMeasure", quantity: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return quantity * 1e3  # convert to cm^-6 pc
