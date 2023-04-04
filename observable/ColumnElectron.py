#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:56:00 2022

@author: alankar
"""

import sys
import numpy as np
from scipy.interpolate import interp1d
# from scipy.integrate import simpson
sys.path.append('..')
sys.path.append('../submodules/AstroPlasma')
from astro_plasma import Ionization
from observable.ColumnDensity import ColumnDensity

class electron_column(ColumnDensity):
    def _additional_fields(self, indx, r_val):
        mode = self.redisProf.ionization
        redshift = self.redisProf.redshift
        num_dens = Ionization().interpolate_num_dens
        
        if not(hasattr(self, 'neHot')):
            self.neHot = np.zeros_like(self.nHhot) 
        if not(hasattr(self, 'neWarm')):
            self.neWarm = np.zeros_like(self.nHWarm) 
        if not(hasattr(self, 'ne')):
            self.ne = np.zeros_like(self.radius)
        
        xh = np.log(self.Temp/(self.THotM(r_val)*np.exp(self.redisProf.sigH**2/2)))
        PvhT = np.exp(-xh**2/(2*self.redisProf.sigH**2))/(self.redisProf.sigH*np.sqrt(2*np.pi))
        xw = np.log(self.Temp/self.redisProf.TmedVW)
        gvwT = self.fvw(r_val)*np.exp(-xw**2/(2*self.redisProf.sigW**2))/(self.redisProf.sigW*np.sqrt(2*np.pi))
        gvhT = np.piecewise(PvhT, [self.Temp>=self.Tcut(r_val),], [lambda xp:xp, lambda xp:0.])
        
        self.neHot[indx,:] =  np.array([num_dens(self.nHhot[indx,i], 
                                                 self.Temp[i], 
                                                 self.metallicity(r_val), redshift, 
                                                 mode=mode, part_type='electron' ) for i in range(self.Temp.shape[0])])
        self.neWarm[indx,:] =  np.array([num_dens(self.nHwarm[indx,i], 
                                                 self.Temp[i], 
                                                 self.metallicity(r_val), redshift, 
                                                 mode=mode, part_type='electron' ) for i in range(self.Temp.shape[0])])
        
        hotInt  = (1-self.fvw(r_val))*np.trapz( (self.neHot[indx,:]*gvhT, xh) ) # global density sensitive     
        warmInt = self.fvw(r_val)*np.trapz( (self.neWarm[indx,:]*gvwT, xw) )
        self.ne[indx] = hotInt + warmInt
        
    def _interpolate_additional_fields(self):
        self.ne = interp1d( self.radius, self.ne, fill_value='extrapolate')
        
    def gen_column(self, b_):
        return super().gen_column(b_, 'ne')
      
