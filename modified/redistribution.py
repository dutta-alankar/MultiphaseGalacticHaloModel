# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:18:19 2023

@author: alankar
"""

import sys
import os
import numpy as np
import pickle
import pathlib
from scipy import interpolate
from scipy import integrate
from scipy.optimize import root
from scipy.special import erf
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any

sys.path.append("..")
from misc.constants import mp, mH, kpc, km, s, K, kB, G, pi, Xp
from misc.coolLambda import cooling_approx
from misc.template import unmodified_field
from abc import ABC, abstractmethod

sys.path.append("..")
sys.path.append("../submodules/AstroPlasma")
from astro_plasma import Ionization
import matplotlib.pyplot as plt

_mpi = True

if _mpi:
    from mpi4py import MPI

    ## start parallel programming ---------------------------------------- #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    # t_start = MPI.Wtime()
else:
    rank = 0
    size = 1

@dataclass
class Redistribution(ABC):
    UNIT_LENGTH = kpc
    UNIT_DENSITY = mp
    UNIT_VELOCITY = km / s
    UNIT_MASS = UNIT_DENSITY * UNIT_LENGTH**3
    UNIT_TIME = UNIT_LENGTH / UNIT_VELOCITY
    UNIT_ENERGY = UNIT_MASS * (UNIT_LENGTH / UNIT_TIME) ** 2
    UNIT_TEMPERATURE = K

    unmodified: unmodified_field
    TmedVW: float = 3.0e5
    sig: float = 0.3  # spread of unmodified temperature redistribution
    cutoff: float = 4.0
    isobaric: Optional[int] = None
    redisType: Optional[str] = None
    _plot_xval: Optional[np.ndarray] = None
    _call_profile: bool = True

    def __post_init__(self: "Redistribution") -> None:
        self.sigH: float = self.sig
        self.sigW: float = self.sig
        self.redshift: float = self.unmodified.redshift
        self.ionization: str = self.unmodified.ionization
        self._required_additional_props()
        self.unmod_filename: str = f"unmod_{self.unmodified._type}_ionization_{self.unmodified.ionization}.pickle"
        self.radius: Optional[np.ndarray] = None
        self.load_unmod: bool = pathlib.Path(self.unmod_filename).is_file()
        if self.redisType is None:
            raise AttributeError("Error: Missing redistribution type specification!")

    @abstractmethod
    def _required_additional_props(self: "Redistribution") -> None:
        pass

    @abstractmethod
    def _calculate_filling_fraction(
        self: "Redistribution", xmin: float, indx: int
    ) -> Tuple[float, float]:
        pass

    def ProfileGen(
        self: "Redistribution",
        radius_: Union[float, int, list[float], list[int], np.ndarray],
    ) -> Tuple:  # takes in radius_ in kpc, returns Halo density and pressure  in CGS
        mu = Ionization.interpolate_mu
        if self.isobaric is None:
            raise AttributeError(
                "Error: Missing thermodynamic modification condition: isobaric/isochoric!"
            )
        isobaric = self.isobaric
        _force_recalc = False

        if type(radius_) == float or type(radius_) == int:
            radius_ = np.array([radius_])
        else:
            radius_ = np.array(radius_)

        if rank == 0:
            display = "Generating" if not(self.load_unmod) else "Loading already saved"
            print(f"{display} {self.unmodified._type} {self.unmodified.ionization} unmodified profile ...", end=" ", flush=True)
        t_start = MPI.Wtime()
        radius = radius_ * kpc / self.UNIT_LENGTH
        if not(self.load_unmod):
            unmod_ret = self.unmodified.ProfileGen(radius_)  # CGS
            # Save the data
            self.unmodified.save()
        else:
            unmod_loaded: Optional[unmodified_field] = None
            if rank == 0:
                with open(self.unmod_filename, "rb") as data_file:
                    unmod_loaded = pickle.load(data_file)               
            unmod_loaded = comm.bcast(unmod_loaded, root=0)
            if unmod_loaded.rho.shape[0] != radius_.shape[0]:
                if rank == 0:
                    print("File loaded has mismatch! Please regenerate unmod profile by deleting the old file", flush=True)
                sys.exit()
            self.unmodified = unmod_loaded
            unmod_ret = (unmod_loaded.rho,
                         unmod_loaded.prsTh,
                         unmod_loaded.prsnTh,
                         unmod_loaded.prsTurb,
                         unmod_loaded.prsTot,
                         unmod_loaded.nH,
                         unmod_loaded.mu,
                        )
        (
            unmod_rho,
            unmod_prsTh,
            _,
            _,
            unmod_prsTot,
            unmod_nH,
            unmod_mu,
        ) = unmod_ret
        t_stop = MPI.Wtime()
        if rank == 0:
            print("Done! Took", (t_stop-t_start), "s", flush=True)
        self.metallicity = self.unmodified.metallicity
        unmod_T = (unmod_prsTh / kB) / (unmod_rho / (unmod_mu * mp))
        self.TmedVH = unmod_T * np.exp(self.sigH**2 / 2)
        self.TmedVu = unmod_T * np.exp(self.sig**2 / 2)
        # This changes even if condition is isobaric, but serves as a good guess for nH
        unmod_prsHTh = unmod_nH * kB * unmod_T

        tdyn = np.sqrt(
            radius**3
            / (
                (
                    G
                    * self.UNIT_LENGTH**2
                    * self.UNIT_DENSITY
                    / self.UNIT_VELOCITY**2
                )
                * (self.unmodified.Halo.Mass(radius_) / self.UNIT_MASS)
            )
        )  # code

        def tcool(ndens, nH, Temp, met):
            return (
                (1.5 + isobaric)
                * ndens
                * kB
                * Temp
                / (nH * nH * cooling_approx(Temp, met))
                / self.UNIT_TIME
            )  # ndens in CGS , tcool in code, 4.34 for FM17 LAMDBA norm

        # Warm gas
        Tstart = 3.9
        Tstop = 6.5
        Temp = np.logspace(Tstart, Tstop, 20 if size<=20 else size)
        # Find tcool/tff for these temperature values
        if self._plot_xval is not None:
            # This mean profile generation was called from plotting method PlotDistributionGen
            Temp = self._plot_xval
        fvw = np.zeros_like(radius)
        fmw = np.zeros_like(radius)
        Tcut = np.zeros_like(radius)
        xmin = np.zeros_like(radius)

        def _calc(
            r_val: float,
        ) -> None:  # This always called for each memeber of radius array
            indx = np.argwhere(np.isclose(radius, r_val))[
                0, 0
            ]  # index for the passed radius
            print_str: str = ""
            if rank == 0:
                print_str = f"{indx+1}/{radius.shape[0]}"
                print(f"{indx+1}/{radius.shape[0]}", end = "\b"*len(print_str), flush=True)

            r_val = r_val * self.UNIT_LENGTH  # CGS
            ndens = unmod_prsTh[indx] / (kB * Temp)  # CGS
            nH_guess = unmod_prsHTh[indx] / (kB * Temp)  # guess

            nH = np.zeros_like(nH_guess)
            for i in range(rank, Temp.shape[0], size):
                nH[i] = 10.0 ** root(
                        lambda LognH: (unmod_prsTh[indx] / kB)
                        * Xp(self.metallicity[indx])
                        * mu(
                            10.0**LognH,
                            Temp[i],
                            self.metallicity[indx],
                            self.redshift,
                            self.ionization,
                        )
                        - 10.0**LognH * Temp[i] * (mH / mp),
                        np.log10(nH_guess[i]),
                    ).x[0]
            if _mpi:
                comm.Barrier()
                # use MPI to get the totals
                _tmp = np.zeros_like(nH)
                comm.Allreduce([nH, MPI.DOUBLE], [_tmp, MPI.DOUBLE], op=MPI.SUM)
                nH = np.copy(_tmp) 
            # if rank == 0:
            #     print("nH =", nH, flush=True)                   

            unmod_tcool = tcool(ndens, nH, Temp, self.metallicity[indx])  # code
            ratio = unmod_tcool / tdyn[indx]

            if self.cutoff >= 0.1:
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff)
                Tcut[indx] = Tmin
                # cutoff in log T where seperation between hot and warm phases occur
                xmin[indx] = np.log(Tmin / (unmod_T[indx] * np.exp(self.sig**2 / 2)))
                # xwarm = np.log(self.TmedVW/(unmod_T[indx]*np.exp(self.sigH**2/2)))
            else:
                # This part is anyways not physically important
                Tmin = interpolate.interp1d(ratio, Temp, fill_value="extrapolate")
                Tmin = Tmin(self.cutoff)
                Tcut[indx] = Tmin
                xmin[indx] = -np.inf

            fmw[indx], fvw[indx] = self._calculate_filling_fraction(xmin[indx], indx)
            if rank == 0:
                print(" "*len(print_str), end = "\b"*len(print_str), flush=True)

        if not(self.load_mod) or _force_recalc:
            t_start = MPI.Wtime()
            if rank == 0:
                print(f"Calculating {self._type} modified profile ...", end=" ", flush=True)
            _ = (*map(_calc, radius),)  # this is faster than just using for loop
            t_stop = MPI.Wtime()
            if rank == 0:
                print("Done! Took", (t_stop-t_start), "s", flush=True)
        else:
            t_start = MPI.Wtime()
            if rank == 0:
                print(f"Loading already saved {self._type} modified profile ...", end=" ", flush=True)
            mod_loaded: Optional[modified_field] = None
            if rank == 0:
                with open(self.mod_filename, "rb") as data_file:
                    mod_loaded = pickle.load(data_file)               
            mod_loaded = comm.bcast(mod_loaded, root=0)
            t_stop = MPI.Wtime()
            if rank == 0:
                print("Done! Took", (t_stop-t_start), "s", flush=True)
                # print(dir(mod_loaded), flush=True)
            self.Tcut = mod_loaded.Tcut
            self.rhowarm_local =  mod_loaded.rhowarm_local
            self.rhohot_local = mod_loaded.rhohot_local
            self.nHwarm_local = mod_loaded.nHwarm_local
            self.nHhot_local = mod_loaded.nHhot_local
            self.mu_warm = mod_loaded.mu_warm
            self.mu_hot = mod_loaded.mu_hot
            self.nwarm_local = mod_loaded.nwarm_local
            self.nhot_local = mod_loaded.nhot_local
            self.nwarm_global = mod_loaded.nwarm_global
            self.nhot_global = mod_loaded.nhot_global
            self.prs_warm = mod_loaded.prs_warm
            self.prs_hot = mod_loaded.prs_hot
            self.radius = mod_loaded.radius
            self.fvw = mod_loaded.fvw
            self.fmw = mod_loaded.fmw
            self.TempDist = mod_loaded.TempDist
            self.xmin = mod_loaded.xmin

            self._plot_xval = mod_loaded._plot_xval
            self._call_profile = mod_loaded._call_profile

        if not(self.load_mod) or _force_recalc:
            t_start = MPI.Wtime()
            if rank == 0:
                print("Doing related calculations ...", end=" ", flush=True)
            # Now calculate the relevant quantities for the modified profile
            self.Tcut = Tcut
            self.rhowarm_local =  np.zeros_like(radius)
            self.rhohot_local = np.zeros_like(radius)
            for indx in range(radius.shape[0]):
                if fvw[indx] <= 0.:
                    self.rhowarm_local[indx] = 0.
                else:
                    self.rhowarm_local[indx] = unmod_rho[indx] * (fmw[indx] / fvw[indx])
                if fvw[indx] >= 1.0:
                    self.rhohot_local[indx] = 0.
                else:
                    self.rhohot_local[indx] = unmod_rho[indx] * ((1 - fmw[indx]) / (1 - fvw[indx]))

            self.nHwarm_local = self.rhowarm_local * Xp(self.metallicity) / mH
            self.nHhot_local = self.rhohot_local * Xp(self.metallicity) / mH

            self.mu_warm = np.zeros_like(radius_)
            for i in range(rank, radius_.shape[0], size):
                self.mu_warm[i] = mu(
                        self.nHwarm_local[i],
                        self.TmedVW * np.exp(-self.sigW**2 / 2),
                        self.metallicity[i],
                        self.redshift,
                        self.ionization,
                    )
            if _mpi:
                comm.Barrier()
                # use MPI to get the totals
                _tmp = np.zeros_like(self.mu_warm)
                comm.Allreduce([self.mu_warm, MPI.DOUBLE], [_tmp, MPI.DOUBLE], op=MPI.SUM)
                self.mu_warm = np.copy(_tmp)  
                
            self.mu_hot = np.zeros_like(radius_)
            for i in range(rank, radius_.shape[0], size):
                self.mu_hot[i] = mu(
                        self.nHhot_local[i],
                        self.TmedVH[i] * np.exp(-self.sigH**2 / 2),
                        self.metallicity[i],
                        self.redshift,
                        self.ionization,
                    )
            if _mpi:
                comm.Barrier()
                # use MPI to get the totals
                _tmp = np.zeros_like(self.mu_hot)
                comm.Allreduce([self.mu_hot, MPI.DOUBLE], [_tmp, MPI.DOUBLE], op=MPI.SUM)
                self.mu_hot = np.copy(_tmp)        

            self.nwarm_local = self.rhowarm_local / (self.mu_warm * mp)
            self.nhot_local = self.rhohot_local / (self.mu_hot * mp)

            self.nwarm_local = np.piecewise(
                self.nwarm_local,
                [
                    np.isnan(self.nwarm_local),
                ],
                [lambda x: 0, lambda x: x],
            )
            self.nhot_local = np.piecewise(
                self.nhot_local,
                [
                    np.isnan(self.nhot_local),
                ],
                [lambda x: 0, lambda x: x],
            )
            self.nwarm_global = self.nwarm_local * fvw
            self.nhot_global = self.nhot_local * (1 - fvw)
            self.prs_warm = (
                self.nwarm_local * kB * self.TmedVW * np.exp(-self.sigW**2 / 2)
            )
            self.prs_hot = self.nhot_local * kB * unmod_T
            self.radius = radius_  # kpc
            self.fvw = fvw
            self.fmw = fmw
            self.TempDist = Temp
            self.xmin = xmin

            self._plot_xval = None
            self._call_profile = False

            t_stop = MPI.Wtime()
            if rank == 0:
                print("Done! Took", (t_stop-t_start), "s", flush=True)

        return (
            self.nhot_local,
            self.nwarm_local,
            self.nhot_global,
            self.nwarm_global,
            self.fvw,
            self.fmw,
            self.prs_hot,
            self.prs_warm,
            self.Tcut,
        )
    
    def save(self: "Redistribution") -> None:
        self._call_profile = False
        if pathlib.Path(self.mod_filename).is_file():
            return
        if rank == 0:
            # print(f"Saving {filename} ...", end=" ", flush=True)
            with open(self.mod_filename, "wb") as f:
                pickle.dump(self, f)
            # print("Done!", flush=True)

    def MassGen(
        self: "Redistribution", radius_: Union[float, list, np.ndarray]
    ) -> Tuple[
        float, float
    ]:  # Takes in r in kpc, returns Halo gas mass of each component in CGS
        if isinstance(radius_, float) or isinstance(radius_, int):
            radius_ = np.array([radius_])
        else:
            radius_ = np.array(radius_)
        if self.radius is None:
            self._call_profile = True
            self.radius = radius_
        else:
            for r_val in radius_:
                if r_val not in self.radius:
                    self._call_profile = True
                    break

        if self._call_profile: # check if profile needs to be calculated
            _ = self.ProfileGen(radius_)
        
        MHot = integrate.cumtrapz(
            4 * pi * (radius_ * kpc) ** 2 * (self.nhot_global * self.mu_hot * mp),
            radius_ * kpc,
        )
        MWarm = integrate.cumtrapz(
            4 * pi * (radius_ * kpc) ** 2 * (self.nwarm_global * self.mu_warm * mp),
            radius_ * kpc,
        )

        return (MHot, MWarm)

    def probability_ditrib_mod(
        self: "Redistribution", *args: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            "ThotM"
            and "fvw"
            and "xmin"
            and "Tcutoff"
            and "Temp" in kwargs
            and len(args) == 1
        ):
            r_val = args[0]
            TmedVH = kwargs["ThotM"](r_val) * np.exp(self.sigH**2 / 2)
            fvw = kwargs["fvw"](r_val)
            Temp = kwargs["Temp"]
            xmin = kwargs["xmin"](r_val)
            Tcutoff = kwargs["Tcutoff"](r_val)
        elif (
            "ThotM"
            and "fvw"
            and "xmin"
            and "Tcutoff"
            and "Temp" in kwargs
            and len(args) == 0
        ):
            TmedVH = kwargs["ThotM"] * np.exp(self.sigH**2 / 2)
            fvw = kwargs["fvw"]
            Temp = kwargs["Temp"]
            xmin = kwargs["xmin"]
            Tcutoff = kwargs["Tcutoff"]
        else:
            TmedVH = self.TmedVH
            fvw = self.fvw
            Temp = self.TempDist
            xmin = self.xmin
            Tcutoff = np.exp(self.xmin) * self.TmedVH

        xh = np.log(Temp / TmedVH)
        gv_unmod = np.exp(-(xh**2) / (2 * self.sigH**2)) / (
            self.sigH * np.sqrt(2 * np.pi)
        )
        xw = np.log(Temp / self.TmedVW)
        gvw = (
            fvw
            * np.exp(-(xw**2) / (2 * self.sigW**2))
            / (self.sigW * np.sqrt(2 * np.pi))
        )
        correction_factor = 0.5 * (1 - erf(xmin / (np.sqrt(2) * self.sigH)))
        gvh = ((1 - fvw) / correction_factor) * np.piecewise(
            gv_unmod,
            [
                Temp >= Tcutoff,
            ],
            [lambda xp: xp, lambda xp: 0.0],
        )

        return (gv_unmod, gvh, gvw)

    def PlotDistributionGen(
        self: "Redistribution", radius: float, figure: Optional[plt.figure] = None
    ) -> None:  # Takes in radius in kpc (one value at any call)
        _save = True
        if self.radius is not None and self._plot_xval is not None:
            if radius in self.radius:
                self._call_profile = False
        if self._call_profile:
            Tstart = 3.9
            Tstop = 7.9
            Temp = np.logspace(Tstart, Tstop, 1000)
            # Find tcool/tff for these temperature values
            self._plot_xval = np.copy(Temp)
            _ = self.ProfileGen(
                np.array(
                    [
                        radius,
                    ]
                )
            )

        if figure is None:
            fig = plt.figure(figsize=(13, 10))
        elif isinstance(figure, plt.Figure):
            fig = figure
        else:
            raise ValueError(f"Invalid value passed to figure = {figure}")
        self.fig = fig
        fig.gca()

        Tcutoff = np.exp(self.xmin) * self.TmedVH
        gv_unmod, gvh, gvw = self.probability_ditrib_mod()

        plt.vlines(
            np.log10(Tcutoff),
            1e-3,
            2.1,
            colors="black",
            linestyles="--",
            label=r"$T_c\ (t_{\rm cool}/t_{\rm ff}=%.1f)$" % self.cutoff,
            linewidth=5,
            zorder=20,
        )
        plt.vlines(
            np.log10(self.TmedVH),
            1e-3,
            2.1,
            colors="tab:red",
            linestyles=":",
            label=r"$T_{med,V}^{(h)}$",
            linewidth=5,
            zorder=30,
        )
        plt.vlines(
            np.log10(self.TmedVW),
            1e-3,
            2.1,
            colors="tab:blue",
            linestyles=":",
            label=r"$T_{med,V}^{(w)}$",
            linewidth=5,
            zorder=40,
        )

        plt.semilogy(
            np.log10(self.TempDist),
            gvh,
            color="tab:red",
            label="hot, modified",
            linewidth=5,
            zorder=5,
        )
        plt.semilogy(
            np.log10(self.TempDist),
            gv_unmod,
            color="tab:red",
            alpha=0.5,
            label="hot, unmodified",
            linewidth=5,
            zorder=6,
        )
        plt.semilogy(
            np.log10(self.TempDist),
            gvw,
            color="tab:blue",
            label="warm",
            linestyle="--",
            linewidth=5,
            zorder=7,
        )

        if self.redisType is not None and _save:
            save_dict = {
                "T_cutoff": Tcutoff,
                "T_hot_M": self.TmedVH * np.exp(-self.sigH**2 / 2),
                "T_med_VW": self.TmedVW,
                "T_hot_u": self.TmedVu * np.exp(-self.sig**2 / 2),
                "TempDist": self.TempDist,
                "gv_unmod": gv_unmod,
                "gv_h": gvh,
                "gv_w": gvw,
                "sig_u": self.sig,
                "sig_H": self.sigH,
                "sig_W": self.sigW,
                "cutoff": self.cutoff,
                "ionization": self.ionization,
                "radius": radius,
            }
            os.system("mkdir -p ./figures")
            with open(
                f"figures/{self.unmodified.unmod_type}_{self.redisType}_{self.ionization}_distrib-r={radius:.1f}kpc.pickle",
                "wb",
            ) as f:
                pickle.dump(save_dict, f)

        if figure is None:
            unmod = (
                "isothermal" if self.unmodified.unmod_type == "isoth" else "isentropic"
            )
            plt.tick_params(
                axis="both", which="major", labelsize=24, direction="out", pad=5
            )
            plt.tick_params(
                axis="both", which="minor", labelsize=24, direction="out", pad=5
            )
            plt.grid()
            plt.title(
                r"$r = $%.1f kpc [%s with isochoric modification] (%s)"
                % (radius, unmod, self.ionization),
                size=28,
            )
            plt.ylim(1e-3, 2.1)
            plt.xlim(5, 7)
            plt.ylabel(r"$T \mathscr{P}_V(T)$", size=28)
            plt.xlabel(r"$\log_{10} (T [K])$", size=28)
            # ax.yaxis.set_ticks_position('both')
            lgd = plt.legend(
                loc="upper right",
                prop={"size": 20},
                framealpha=0.3,
                shadow=False,
                fancybox=True,
                bbox_to_anchor=(1.1, 1),
            )
            plt.savefig(
                "isothermal_isochoric_PDF_%s.png" % self.ionization,
                transparent=True,
                bbox_inches="tight",
                bbox_extra_artists=(lgd,),
            )
            plt.show()
