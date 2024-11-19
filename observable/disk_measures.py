# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:58:36 2022

@author: alankar

Reference: Yamasaki and Totani 2020
"""

import sys

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
import numpy as np
from typing import Union, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from misc.ProgressBar import ProgressBar
import observable.CoordinateTrans as transform
import observable.maps as maps
from astro_plasma import Ionization

@dataclass
class Disk_profile:
    nH0: float = 4.8e-3  # cm^-3
    R0: float = 8.5  # kpc
    z0: float = 3.0  # kpc
    metallicity: float = 1.0
    redshift: float = 0.0
    mode: str = "PIE"

    def __post_init__(self: "Disk_profile") -> None:
        self.TDisk: float = 1.5e6  # K
        self.nH: Callable = lambda R, z: self.nH0 * np.exp(
            -(R / self.R0 + np.fabs(z) / self.z0)
        )
        self.nH_block: Callable = lambda R, z: np.piecewise( R, [np.logical_and(R<self.R0, np.abs(z)<0.5*self.z0),], [self.nH0, 0.] )
        num_dens = Ionization.interpolate_num_dens
        self.ne0: float = num_dens(
            self.nH0, self.TDisk, self.metallicity, self.redshift, self.mode, "electron"
        )
        self.ni0: float = num_dens(
            self.nH0, self.TDisk, self.metallicity, self.redshift, self.mode, "ion"
        )

class MeasuresDisk(maps.MapInit, ABC):
    def __init__(self: "MeasuresDisk", rvir):
        super().__init__(rvir=rvir)
        self._field_func = self._measure_field()

    @abstractmethod
    def _measure_field(self: "MeasuresDisk") -> Callable:
        pass

    def set_disk(self: "MeasuresDisk", 
                 nH0: float = 4.8e-3,  # cm^-3
                 R0: float = 8.5,  # kpc
                 z0: float = 3.0,  # kpc
                 metallicity: float = 1.0,
                 redshift: float = 0.0,
                 mode: str = "PIE",
                 ) -> None:
        self.disk = Disk_profile(nH0=nH0, 
                                 R0=R0, 
                                 z0=z0,
                                 metallicity=metallicity, 
                                 redshift=redshift, 
                                 mode=mode)

    def make_map(
        self: "MeasuresDisk",
        l: Union[float, list, np.ndarray],
        b: Union[float, list, np.ndarray],
        showProgress: Optional[bool] = True,
    ) -> np.ndarray:
        l, b = super().prepare(l, b)

        field = np.zeros_like(l)
        if isinstance(l, np.ndarray):
            if showProgress:
                progBar = None

                for i in range(field.shape[0]):
                    for j in range(field.shape[1]):
                        # 1.0e-3 is needed to resolve the disk
                        LOSsample = np.logspace(
                            np.log10(1.0e-3 * self.integrateTill[i, j]),
                            np.log10(self.integrateTill[i, j]),
                            100,
                        )
                        radius, phi, theta = transform.toGalC(
                            np.array(l)[i, j], np.array(b)[i, j], LOSsample
                        )
                        height = np.abs(radius * np.cos(np.deg2rad(theta)))
                        radius = np.abs(
                            radius * np.sin(np.deg2rad(theta))
                        )  # np.cos(np.deg2rad(phi))) # along disk

                        val = np.trapz(
                            np.nan_to_num(self._field_func(radius, height)), LOSsample
                        )

                        field[i, j] = val  # CGS unit * kpc
                        if i == 0 and j == 0:
                            progBar = ProgressBar()
                        progBar.progress(
                            i * field.shape[1] + j + 1, field.shape[0] * field.shape[1]
                        )
                progBar.end()
            else:
                def _calc(tup):
                    l_val, b_val, _integrateTill = tup
                    # print(l_val, b_val, _integrateTill)
                    LOSsample = np.logspace(
                        np.log10(1.0e-3 * _integrateTill), np.log10(_integrateTill), 100
                    )
                    radius, phi, theta = transform.toGalC(l_val, b_val, LOSsample)
                    height = np.abs(radius * np.cos(np.deg2rad(theta)))
                    radius = np.abs(
                        radius * np.sin(np.deg2rad(theta))
                    )  # np.cos(np.deg2rad(phi))) #along disk
                    field = np.trapz(
                        np.nan_to_num(self._field_func(radius, height)), LOSsample
                    )
                    return field  # CGS kpc

                tup = (
                    *zip(
                        np.array(l).flatten(),
                        np.array(b).flatten(),
                        self.integrateTill.flatten(),
                    ),
                )
                field = np.array((*map(_calc, tup),)).reshape(l.shape)  # CGS kpc
        else:
            LOSsample = np.logspace(
                np.log10(1.0e-3 * self.integrateTill), np.log10(self.integrateTill), 100
            )
            radius, phi, theta = transform.toGalC(l, b, LOSsample)
            height = np.abs(radius * np.cos(np.deg2rad(theta)))
            radius = np.abs(
                radius * np.sin(np.deg2rad(theta))
            )  # np.cos(np.deg2rad(phi))) #along disk
            field = np.trapz(
                np.nan_to_num(self._field_func(radius, height)), LOSsample
            )  # CGS kpc

        field *= 1e3  # convert to CGS pc
        return field



class DiskDM(MeasuresDisk):
    def _measure_field(self: "DiskDM") -> Callable:
        disk = self.disk if hasattr(self, "disk") else Disk_profile()
        ne_prof = lambda R, z: (disk.ne0 / disk.nH0) * disk.nH(R, z)
        return ne_prof


class DiskEM(MeasuresDisk):
    def _measure_field(self: "DiskEM") -> Callable:
        disk = self.disk if hasattr(self, "disk") else Disk_profile()
        nH_prof = lambda R, z:  disk.nH(R, z)
        ne_prof = lambda R, z: (disk.ne0 / disk.nH0) * disk.nH(R, z)
        return lambda R, z: ne_prof(R, z) * nH_prof(R, z)
