#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:22:15 2023

@author: alankar
"""

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../submodules/AstroPlasma')
import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.optimize import root
from misc.ProgressBar import ProgressBar
from  misc.constants import mp, mH, kpc, kB, Xp
import observable.CoordinateTrans as transform
import observable.maps as maps
from misc.template import modified_field
from typing import Union, Optional, List, Callable
from astro_plasma import Ionization

class Measure(maps.MapInit, ABC):
    
    def __init__(self, redisProf: modified_field):
        super().__init__(redisProf)
        self.redisProf = redisProf
        self.genProf = False
        
    def _generate_measurable(self: "Measure", 
               distance: Union[float, int, np.ndarray]) -> tuple :
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        
        num_dens = lambda tup: Ionization.interpolate_num_dens(*tup)
        
        # redistributed profile is generated for only a limited number of points and used for interpolation
        if not(self.genProf):
            rend = 1.01*self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.__class__.UNIT_LENGTH/kpc)
            print("Doing one time profile calculation", flush=True)
            radius_ = np.logspace(np.log10(5.0), np.log10(rend),20) # kpc
            nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = self.redisProf.ProfileGen(radius_)
            self.genProf = True
            print('Complete!', flush=True)
            
        mu = Ionization.interpolate_mu
        
        nHhot_local  = interp1d( self.redisProf.radius, self.redisProf.nHhot_local, fill_value='extrapolate')
        nHwarm_local = interp1d( self.redisProf.radius, self.redisProf.nHwarm_local, fill_value='extrapolate')
        prs_hot      = interp1d( self.redisProf.radius, self.redisProf.prs_hot, fill_value='extrapolate')
        prs_warm     = interp1d( self.redisProf.radius, self.redisProf.prs_warm, fill_value='extrapolate')
        Tcut         = interp1d( self.redisProf.radius, self.redisProf.Tcut, fill_value='extrapolate')
        metallicity  = interp1d( self.redisProf.radius, self.redisProf.unmodified.metallicity, fill_value='extrapolate')
        fvw          = interp1d( self.redisProf.radius, self.redisProf.fvw, fill_value='extrapolate')
        TmedVH       = interp1d( self.redisProf.radius, self.redisProf.TmedVH, fill_value='extrapolate')
                
        Temp  = self.redisProf.TempDist
        THotM  = interp1d( self.redisProf.radius, (self.redisProf.prs_hot/(self.redisProf.nhot_local*kB)), fill_value='extrapolate')
        
        def _weighted_avg_quantity(r_val: float, 
                                   nHhot: np.ndarray, 
                                   nHwarm: np.ndarray, 
                                   gvhT: np.ndarray, xh: np.ndarray, 
                                   gvwT: np.ndarray, xw: np.ndarray, 
                                   part_type: str, ) -> float:
            tup = (*zip( nHhot, Temp, 
                         metallicity(r_val)*np.ones(Temp.shape[0]), 
                         redshift*np.ones(Temp.shape[0]),
                         [mode,]*Temp.shape[0], [part_type,]*Temp.shape[0] ),)
            
            quanHot = np.array( (*map(num_dens, tup),) ).reshape(Temp.shape)
            
            tup = (*zip( nHwarm, Temp, 
                         metallicity(r_val)*np.ones(Temp.shape[0]), 
                         redshift*np.ones(Temp.shape[0]),
                         [mode,]*Temp.shape[0], [part_type,]*Temp.shape[0] ),)
            
            quanWarm = np.array( (*map(num_dens, tup),) ).reshape(Temp.shape)
            
            hotInt  = (1-fvw(r_val))*np.trapz( quanHot*gvhT, xh ) #global density sensitive, extra filling factor for global 
        
            warmInt = fvw(r_val)*np.trapz( quanWarm*gvwT, xw )
            
            return (hotInt + warmInt)
        
        def _calc(r_val: Union[float, int]) -> np.ndarray:    
            xh = np.log(Temp/(THotM(r_val)*np.exp(self.redisProf.sigH**2/2)))
            PvhT = np.exp(-xh**2/(2*self.redisProf.sigH**2))/(self.redisProf.sigH*np.sqrt(2*np.pi))
            xw = np.log(Temp/self.redisProf.TmedVW)
            gvwT = fvw(r_val)*np.exp(-xw**2/(2*self.redisProf.sigW**2))/(self.redisProf.sigW*np.sqrt(2*np.pi))
            gvhT = np.piecewise(PvhT, [Temp>=Tcut(r_val),], [lambda xp:xp, lambda xp:0.])
            
            # Approximation is that nH T is also constant like n T used as guess
            nHhot_guess  = nHhot_local(r_val)*TmedVH(r_val)*np.exp(-self.redisProf.sigH**2/2)/Temp # CGS
            nHwarm_guess = nHwarm_local(r_val)*self.redisProf.TmedVW*np.exp(-self.redisProf.sigW**2/2)/Temp # CGS
            
            nHhot  = 10.**np.array([root( lambda LognH: (prs_hot(r_val)/(kB*Temp[i])) * Xp(metallicity(r_val))*\
                                    mu(10**LognH, Temp[i], metallicity(r_val), redshift, mode) - \
                                    (mH/mp)*(10**LognH), np.log10(nHhot_guess[i]) ).x[0] for i in range(Temp.shape[0])])
            nHwarm = 10.**np.array([root( lambda LognH: (prs_warm(r_val)/(kB*Temp[i])) * Xp(metallicity(r_val))*\
                                    mu(10**LognH, Temp[i], metallicity(r_val), redshift, mode) - \
                                    (mH/mp)*(10**LognH), np.log10(nHwarm_guess[i]) ).x[0] for i in range(Temp.shape[0])])
                          
            ne = _weighted_avg_quantity(r_val,
                                        nHhot, nHwarm, 
                                        gvhT, xh, 
                                        gvwT, xw, 
                                        "electron" )
            ni = _weighted_avg_quantity(r_val,
                                        nHhot, nHwarm, 
                                        gvhT, xh, 
                                        gvwT, xw, 
                                        "ion" )
            
            return np.array([ne, ni])
        
        _quan = np.array([*map(_calc, distance)])
        ne, ni = _quan[:,0], _quan[:,1]
        return (ne, ni)
    
    @abstractmethod
    def observable_quantity(self: "Measure", 
                            funcs: List[Callable], 
                            distance: Union[float, list, np.ndarray],
                            LOS_sample: Union[list, np.ndarray],) -> float:
        pass
    
    @abstractmethod
    def post_process_observable(self: "Measure", 
                                quantity: Union[float, np.ndarray]) -> np.ndarray:
        pass
            
    def make_map(self: "Measure", 
                 l: Union[float, list, np.ndarray], 
                 b: Union[float, list, np.ndarray], 
                 showProgress: Optional[bool] = True) -> Union[float, np.ndarray]:
        l, b = super().prepare(l,b)
        rend = self.redisProf.unmodified.rCGM*(self.redisProf.unmodified.UNIT_LENGTH/kpc)
        
        distance = np.logspace(np.log10(5.0), 1.01*np.log10(rend), 20) # kpc
        print('Generating profiles ...')
        ne_prof, ni_prof = self._generate_measurable(distance)
        print('Complete!')

        ne_prof  = interp1d( np.log10(distance), np.log10(ne_prof), fill_value='extrapolate')
        ni_prof  = interp1d( np.log10(distance), np.log10(ni_prof), fill_value='extrapolate')
        
        if isinstance(l, np.ndarray):          
            if showProgress: 
                progBar = None
                self.observable = np.zeros_like(l)
                for i in range(self.observable.shape[0]):
                    for j in range(self.observable.shape[1]):
                        LOSsample = np.logspace(np.log10(1e-3*self.integrateTill[i,j]), np.log10(self.integrateTill[i,j]), 100) # points on the LOS
                        radius, phi, theta = transform.toGalC(l[i,j], b[i,j], LOSsample)
                        height = np.abs(radius*np.cos(np.deg2rad(theta)))
                        radius = np.abs(radius*np.sin(np.deg2rad(theta))) 
                        distance = np.sqrt(radius**2 + height**2)
                        observable = self.observable_quantity([ne_prof, ni_prof], distance, LOSsample)
                        # np.trapz( np.nan_to_num(10.**ne_prof(np.log10(distance))) , LOSsample)
                        self.observable[i,j] =  observable # cm^-3 kpc
                        if (i==0 and j==0): progBar = ProgressBar()
                        progBar.progress(i*self.observable.shape[1]+j+1, 
                                         self.observable.shape[0]*self.observable.shape[1])
                progBar.end()
            else:
                def _calc(tup):
                    l_val, b_val, integrateTill = tup
                    LOSsample = np.logspace(np.log10(1e-3*integrateTill), np.log10(integrateTill), 100) 
                    radius, phi, theta = transform.toGalC(l_val, b_val, LOSsample)
                    height = np.abs(radius*np.cos(np.deg2rad(theta)))
                    radius = np.abs(radius*np.sin(np.deg2rad(theta))) # along disk
                    distance = np.sqrt(radius**2 + height**2)
                    observable = self.observable_quantity([ne_prof, ni_prof], distance, LOSsample)
                    return observable # cm^-3 kpc
                    
                tup = (*zip(l.flatten(), b.flatten(), self.integrateTill.flatten()),)
                self.observable = np.array( (*map(_calc, tup),) ).reshape(l.shape) # cm^-6 kpc
        else:
            LOSsample = np.logspace(np.log10(1e-3*self.integrateTill), np.log10(self.integrateTill), 100)
            radius, phi, theta = transform.toGalC(l, b, LOSsample)
            height = np.abs(radius*np.cos(np.deg2rad(theta)))
            radius = np.abs(radius*np.sin(np.deg2rad(theta))) # along disk
            distance = np.sqrt(radius**2 + height**2)
            self.observable = self.observable_quantity([ne_prof, ni_prof], distance, LOSsample)
        
        self.observable = self.post_process_observable(self.observable)
        return self.observable
