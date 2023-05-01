# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:22:15 2023

@author: alankar
"""

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../submodules/AstroPlasma")
import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.optimize import root
from misc.ProgressBar import ProgressBar
from misc.constants import mp, mH, kpc, kB, Xp
import observable.CoordinateTrans as transform
import observable.maps as maps
from misc.template import modified_field
from typing import Union, Optional, List, Callable
from astro_plasma import Ionization
from observable.internal_interpolation import _interpolate_internal_variables


class Measure(maps.MapInit, _interpolate_internal_variables, ABC):
    def __init__(self, redisProf: modified_field):
        super().__init__(redisProf)
        self.redisProf = redisProf
        self.observable: Optional[Union[float, int, np.ndarray]] = None

    def _generate_measurable(
        self: "Measure", distance: Union[float, int, np.ndarray]
    ) -> tuple:
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift

        _ = self._interpolate_vars()
        mu = Ionization.interpolate_mu
        num_dens = lambda tup: Ionization.interpolate_num_dens(*tup)

        def _weighted_avg_quantity(
            r_val: float,
            nHhot: np.ndarray,
            nHwarm: np.ndarray,
            gvh: np.ndarray,
            xh: np.ndarray,
            gvw: np.ndarray,
            xw: np.ndarray,
            part_types: list[str],
        ) -> float:
            quanHot = np.zeros((len(part_types), self.Temp.shape[0]))
            quanWarm = np.zeros_like(quanHot)
            for indx, part_type in enumerate(part_types):
                tup = (
                    *zip(
                        nHhot,
                        self.Temp,
                        self.metallicity(r_val) * np.ones(self.Temp.shape[0]),
                        redshift * np.ones(self.Temp.shape[0]),
                        [
                            mode,
                        ]
                        * self.Temp.shape[0],
                        [
                            part_type,
                        ]
                        * self.Temp.shape[0],
                    ),
                )

                quanHot[indx, :] = np.array((*map(num_dens, tup),)).reshape(
                    self.Temp.shape
                )

                tup = (
                    *zip(
                        nHwarm,
                        self.Temp,
                        self.metallicity(r_val) * np.ones(self.Temp.shape[0]),
                        redshift * np.ones(self.Temp.shape[0]),
                        [
                            mode,
                        ]
                        * self.Temp.shape[0],
                        [
                            part_type,
                        ]
                        * self.Temp.shape[0],
                    ),
                )

                quanWarm[indx, :] = np.array((*map(num_dens, tup),)).reshape(
                    self.Temp.shape
                )

            hotInt = (1 - self.fvw(r_val)) * np.trapz(
                np.product(quanHot, axis=0) * gvh, xh
            )
            # global density sensitive, extra filling factor for global

            warmInt = self.fvw(r_val) * np.trapz(np.product(quanWarm, axis=0) * gvw, xw)

            return hotInt + warmInt

        def _calc(r_val: Union[float, int]) -> np.ndarray:
            _, gvh, gvw = self.redisProf.probability_ditrib_mod(
                r_val,
                ThotM=self.ThotM,
                fvw=self.fvw,
                Temp=self.Temp,
                xmin=self.xmin,
                Tcutoff=self.Tcut,
            )
            TmedVH = self.ThotM(r_val) * np.exp(self.redisProf.sigH**2 / 2)
            xh = np.log(self.Temp / TmedVH)
            xw = np.log(self.Temp / self.redisProf.TmedVW)

            # Approximation is that nH T is also constant like n T used as guess
            nHhot_guess = (
                self.nHhot_local(r_val)
                * TmedVH
                * np.exp(-self.redisProf.sigH**2 / 2)
                / self.Temp
            )  # CGS
            nHwarm_guess = (
                self.nHwarm_local(r_val)
                * self.redisProf.TmedVW
                * np.exp(-self.redisProf.sigW**2 / 2)
                / self.Temp
            )  # CGS

            nHhot = 10.0 ** np.array(
                [
                    root(
                        lambda LognH: (self.prs_hot(r_val) / (kB * self.Temp[i]))
                        * Xp(self.metallicity(r_val))
                        * mu(
                            10**LognH,
                            self.Temp[i],
                            self.metallicity(r_val),
                            redshift,
                            mode,
                        )
                        - (mH / mp) * (10**LognH),
                        np.log10(nHhot_guess[i]),
                    ).x[0]
                    for i in range(self.Temp.shape[0])
                ]
            )
            nHwarm = 10.0 ** np.array(
                [
                    root(
                        lambda LognH: (self.prs_warm(r_val) / (kB * self.Temp[i]))
                        * Xp(self.metallicity(r_val))
                        * mu(
                            10**LognH,
                            self.Temp[i],
                            self.metallicity(r_val),
                            redshift,
                            mode,
                        )
                        - (mH / mp) * (10**LognH),
                        np.log10(nHwarm_guess[i]),
                    ).x[0]
                    for i in range(self.Temp.shape[0])
                ]
            )

            ne = _weighted_avg_quantity(
                r_val,
                nHhot,
                nHwarm,
                gvh,
                xh,
                gvw,
                xw,
                [
                    "electron",
                ],
            )
            neni = _weighted_avg_quantity(
                r_val, nHhot, nHwarm, gvh, xh, gvw, xw, ["electron", "ion"]
            )

            return np.array([ne, neni])

        _quan = np.array([*map(_calc, np.array(distance))])
        ne, neni = _quan[:, 0], _quan[:, 1]
        return (ne, neni)

    @abstractmethod
    def observable_quantity(
        self: "Measure",
        funcs: List[Callable],
        distance: Union[list, np.ndarray],
        LOS_sample: Union[list, np.ndarray],
    ) -> float:
        pass

    @abstractmethod
    def post_process_observable(
        self: "Measure", quantity: Union[float, np.ndarray]
    ) -> Union[float, int, np.ndarray]:
        pass

    def make_map(
        self: "Measure",
        l: Union[float, list, np.ndarray],
        b: Union[float, list, np.ndarray],
        showProgress: Optional[bool] = True,
    ) -> Union[float, int, np.ndarray]:
        l, b = super().prepare(l, b)
        if self.integrateTill is None:
            raise ValueError("Error: Trouble with integral limit generation!")
        rend = self.redisProf.unmodified.rCGM * (
            self.redisProf.unmodified.UNIT_LENGTH / kpc
        )

        distance = np.logspace(np.log10(5.0), 1.01 * np.log10(rend), 20)  # kpc
        print("Generating profiles ...")
        ne_val, neni_val = self._generate_measurable(distance)
        print("Complete!")

        ne_prof = interp1d(
            np.log10(distance), np.log10(ne_val), fill_value="extrapolate"
        )
        neni_prof = interp1d(
            np.log10(distance), np.log10(neni_val), fill_value="extrapolate"
        )

        if isinstance(l, np.ndarray):
            if showProgress:
                progBar = None
                self.observable = np.zeros_like(l)
                for i in range(self.observable.shape[0]):
                    for j in range(self.observable.shape[1]):
                        LOSsample = np.logspace(
                            np.log10(1e-3 * self.integrateTill[i, j]),
                            np.log10(self.integrateTill[i, j]),
                            100,
                        )  # points on the LOS
                        radius, phi, theta = transform.toGalC(
                            np.array(l)[i, j], np.array(b)[i, j], LOSsample
                        )
                        height = np.abs(radius * np.cos(np.deg2rad(theta)))
                        radius = np.abs(radius * np.sin(np.deg2rad(theta)))
                        distance = np.sqrt(radius**2 + height**2)
                        observable = self.observable_quantity(
                            [ne_prof, neni_prof], distance, LOSsample
                        )
                        # np.trapz( np.nan_to_num(10.**ne_prof(np.log10(distance))) , LOSsample)
                        self.observable[i, j] = observable  # cm^-3 kpc
                        if i == 0 and j == 0:
                            progBar = ProgressBar()
                        progBar.progress(
                            i * self.observable.shape[1] + j + 1,
                            self.observable.shape[0] * self.observable.shape[1],
                        )
                progBar.end()
            else:

                def _calc(tup):
                    l_val, b_val, integrateTill = tup
                    LOSsample = np.logspace(
                        np.log10(1e-3 * integrateTill), np.log10(integrateTill), 100
                    )
                    radius, phi, theta = transform.toGalC(l_val, b_val, LOSsample)
                    height = np.abs(radius * np.cos(np.deg2rad(theta)))
                    radius = np.abs(radius * np.sin(np.deg2rad(theta)))  # along disk
                    distance = np.sqrt(radius**2 + height**2)
                    observable = self.observable_quantity(
                        [ne_prof, neni_prof], distance, LOSsample
                    )
                    return observable  # cm^-3 kpc

                tup = (
                    *zip(
                        np.array(l).flatten(),
                        np.array(b).flatten(),
                        self.integrateTill.flatten(),
                    ),
                )
                self.observable = np.array((*map(_calc, tup),)).reshape(
                    l.shape
                )  # cm^-6 kpc
        else:
            LOSsample = np.logspace(
                np.log10(1e-3 * self.integrateTill), np.log10(self.integrateTill), 100
            )
            radius, phi, theta = transform.toGalC(l, b, LOSsample)
            height = np.abs(radius * np.cos(np.deg2rad(theta)))
            radius = np.abs(radius * np.sin(np.deg2rad(theta)))  # along disk
            distance = np.sqrt(radius**2 + height**2)
            self.observable = self.observable_quantity(
                [ne_prof, neni_prof], distance, LOSsample
            )

        self.observable = self.post_process_observable(self.observable)
        if self.observable is None:
            raise ValueError("Error: Trouble generating desired field!")
        return self.observable
