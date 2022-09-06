# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:53:50 2022

@author: Alankar
"""

import sys
import numpy as np
from scipy import interpolate
from scipy import integrate
sys.path.append('..')
from misc.constants import *

def ColumnDensityGen(b_, met, redisProf, element=8, ion=6): #takes in b_ in kpc, returns col dens in CGS
    
    if isinstance(b_, float) or isinstance(b_, int):
        b_ = np.array([b_])
    
    frac = np.loadtxt('ion-frac-Oxygen.txt', skiprows=1, 
                      converters={i+1: lambda x: -np.inf if x==b'--' else x for i in range(element+1)})
    fOVII = interpolate.interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 
    
    radius = np.linspace(b_[0]-0.1, redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, 400)
    _, _, nhot_global, nwarm_global, fvw, fmw, _, _, Tcut = redisProf.ProfileGen(radius)
    
    #fvh = 1-fvw
    #fmh = 1-fmw
    
    unmod_rho, unmod_prsTh, _, _, _ = redisProf.unmodified.ProfileGen(radius) #CGS
    unmod_n = unmod_rho/(mu*mp) #CGS
    unmod_T = unmod_prsTh/(unmod_n*kB) # mass avg temperature for unmodified profile
    
    Tstart = 4.5
    Tstop  = 7.5
    Temp  = np.logspace(Tstart, Tstop, 400)
    
    x = np.log(Temp/(unmod_T*np.exp(redisProf.sigH**2/2)))
    PvhT = np.exp(-x**2/(2*redisProf.sigH**2))/(redisProf.sigH*np.sqrt(2*pi))
    xp = np.log(Temp/redisProf.TmedVW)
    PvwT = fvw*np.exp(-xp**2/(2*redisProf.sigW**2))/(redisProf.sigW*np.sqrt(2*pi))
    PvhT = np.piecewise(PvhT, [Temp>=Tcut,], [lambda xpp:xpp, lambda xpp:0.])
    
    PvwT = interpolate.interp1d(np.log10(Temp), PvwT)
    PvhT = interpolate.interp1d(np.log10(Temp), PvhT)
    
    a0 = 4.9e-4 # Asplund et al. 2009
    
    nOVII = np.zeros_like(radius)
    for indx, r_val in enumerate(radius) :
        nOVII[indx] = a0*met*(mu/muHp)*(
               nhot_global[indx] *np.trapz(PvhT(np.log10(Temp))*10**fOVII(np.log10(Temp)), np.log10(Temp)) #integrate.quad(lambda T: (10**fOVII(T))*PvhT(T), 4.6, 7.4)[0]  
             + nwarm_global[indx]*np.trapz(PvwT(np.log10(Temp))*10**fOVII(np.log10(Temp)), np.log10(Temp)) #integrate.quad(lambda T: (10**fOVII(T))*PvwT(T), 4.6, 7.4)[0]
              )
    nOVII = interpolate.interp1d(radius, nOVII, fill_value="extrapolate") #CGS
    
    coldensOVII = np.zeros_like(b_)
    
    for indx, b_val in enumerate(b_):
        coldensOVII[indx] = 2*integrate.quad(lambda r: nOVII(r)*r/np.sqrt(r**2-b_val**2), 
                                b_val, redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc)[0] #kpc cm-3
    
    if len(b_) == 1: return coldensOVII[0]*kpc
    else: return coldensOVII*kpc