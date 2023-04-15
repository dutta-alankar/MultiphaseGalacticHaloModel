# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:31:46 2021

@author: alankar
"""

import numpy as np
from misc.template import modified_field
from typing import Union, Optional
from misc.constants import kpc


class MapInit(object):
    def __init__(self: "MapInit", redisProf: "modified_field") -> None:
        self.redisProf = redisProf
        self.integrateTill: Optional[np.ndarray] = None

    def prepare(
        self: "MapInit",
        l: Union[float, int, list, np.ndarray],
        b: Union[float, int, list, np.ndarray],
    ) -> tuple:
        L = np.array(l)
        B = np.array(b)

        R200 = (
            self.redisProf.unmodified.Halo.r200
            * (self.redisProf.unmodified.Halo.UNIT_LENGTH / kpc)
            * np.ones_like(L)
        )
        if L.ndim == 1 or B.ndim == 1:
            L, B = np.meshgrid(np.array(L), np.array(B))

        SuntoGC = 8.0  # kpc
        costheta = np.cos(np.deg2rad(L)) * np.cos(np.deg2rad(B))

        root1 = np.abs(
            SuntoGC
            * costheta
            * (1 + np.sqrt(1 + (R200**2 - SuntoGC**2) / (SuntoGC * costheta) ** 2))
        )
        root2 = np.abs(
            SuntoGC
            * costheta
            * (1 - np.sqrt(1 + (R200**2 - SuntoGC**2) / (SuntoGC * costheta) ** 2))
        )
        root_large = np.select([root1 > root2, root1 < root2], [root1, root2])
        root_small = np.select([root1 < root2, root1 > root2], [root1, root2])
        # True if both roots are real
        # real  = np.logical_not( np.logical_and(np.iscomplex(root1), np.iscomplex(root2)) )
        B_cond = np.logical_and(B > -90, B < 90)
        L_cond = np.logical_or(
            np.logical_and(L > 0, L < 90), np.logical_and(L > 270, L < 360)
        )
        large_root_select = np.logical_and(B_cond, L_cond)

        self.integrateTill = np.select(
            [large_root_select, np.logical_not(large_root_select)],
            [root_large, root_small],
            default=R200,
        )

        return (L, B)
