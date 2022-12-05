#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:51:35 2022

@author: alankar
"""

import sys
import numpy as np
from scipy import interpolate
from scipy import integrate
sys.path.append('..')
from misc.constants import *
from misc.emission import interpolate_emission

def _fourPiNujNu_cloudy( ndens, temperature, metallicity, redshift, indx=3000, cleanup=False):
    import time
    import os
    mode = 'PIE'
    X_solar = 0.7154
    Y_solar = 0.2703
    Z_solar = 0.0143
    
    Xp = X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
    Yp = Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
    Zp = metallicity*Z_solar # completely ionized plasma; Z varied independent of nH and nHe; Metals==Oxygen
    
    mu   = 1./(2*Xp+0.75*Yp+0.5625*Zp)
    mup  = 1./(2*Xp+0.75*Yp+(9./16.)*Zp)
    muHp = 1./Xp
    mue  = 2./(1+Xp)
    mui  = 1./(1/mu-1/mue)
      
    nH    = (mu/muHp)*ndens # cm^-3

    background = f"table HM12 redshift {redshift:.2f}" if mode=='PIE' else '\n'
    stream = \
    f"""
# ----------- Auto generated from generateCloudyScript.py ----------------- 
#Created on {time.ctime()}
#
#@author: alankar
#
CMB redshift {redshift:.2f}
{background}
sphere
radius 150 to 151 linear kiloparsec
##table ISM
abundances "solar_GASS10.abn"
metals {metallicity:.2e} linear
hden {np.log10(nH):.2f} log
constant temperature, T={temperature:.2e} K linear
stop zone 1
iterate convergence
age 1e9 years
##save continuum "spectra.lin" units keV no isotropic
save diffuse continuum "{"emission_%s.lin"%mode if indx==None else "emission_%s_%09d.lin"%(mode,indx)}" units keV 
    """
    stream = stream[1:]
    
    if not(os.path.exists('./auto')): 
        os.system('mkdir -p ./auto')
    if not(os.path.isfile('./auto/cloudy.exe')): 
        os.system('cp ./cloudy.exe ./auto')
    if not(os.path.isfile('./auto/libcloudy.so')): 
        os.system('cp ./libcloudy.so ./auto')
        
    filename = "autoGenScript_%s_%09d.in"%(mode,indx)
    with open("./auto/%s"%filename, "w") as text_file:
        text_file.write(stream)
        
    os.system("cd ./auto && ./cloudy.exe -r %s"%filename[:-3])
    
    data = np.loadtxt("./auto/emission_%s.lin"%mode if indx==None else "./auto/emission_%s_%09d.lin"%(mode,indx) )
    if cleanup: 
        os.system('rm -rf ./auto/')
    
    return np.vstack( (data[:,0],data[:,-1])  ).T

def LuminositySpectrumGen(redisProf, redshift=0.001): #, mode='CIE'): 
    
    _do_unmodified = True
    _use_cloudy = True
    
    EmmSpectrum = interpolate_emission()
    fourPiNujNu = EmmSpectrum.interpolate
    
    radius = np.linspace(redisProf.unmodified.Halo.r0*redisProf.unmodified.Halo.UNIT_LENGTH/kpc, 
                         redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, 200)    
    
    unmod_rho, unmod_prsTh, _, _, _ = redisProf.unmodified.ProfileGen(radius) #CGS
    metallicity = redisProf.unmodified.metallicity
    
    X_solar = 0.7154
    Y_solar = 0.2703
    Z_solar = 0.0143
    
    Xp = X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
    Yp = Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
    Zp = metallicity*Z_solar # completely ionized plasma; Z varied independent of nH and nHe; Metals==Oxygen
    
    mu   = 1./(2*Xp+0.75*Yp+0.5625*Zp)
    mup  = 1./(2*Xp+0.75*Yp+(9./16.)*Zp)
    muHp = 1./Xp
    mue  = 2./(1+Xp)
    mui  = 1./(1/mu-1/mue)
    
    unmod_n = unmod_rho/(mu*mp) #CGS
    unmod_T = unmod_prsTh/(unmod_n*kB) # mass avg temperature for unmodified profile
    
    Tstart = 4.5
    Tstop  = 7.5
    Temp  = np.logspace(Tstart, Tstop, 10)
    
    luminosityTot = np.zeros_like(EmmSpectrum.energy)
    
    if (_do_unmodified and not(_use_cloudy)):
         for indx, r_val in enumerate(radius) :
             rad  = r_val*kpc #0.5*((radius[indx]+radius[indx-1]) if indx!=0 else 0.5*radius[indx])*kpc
             dr_val = (radius[-1]-radius[-2]) if indx==len(radius)-1 else (radius[indx+1]-radius[indx])*kpc
             
             luminosityTot += (fourPiNujNu (unmod_n[indx], unmod_T[indx], metallicity[indx], redshift)[:,1]) * (4*np.pi) * dr_val * rad**2
                                        
         return np.vstack((EmmSpectrum.energy, luminosityTot/EmmSpectrum.energy)).T
     
    if (_do_unmodified and (_use_cloudy)):
        energy = None
        for indx, r_val in enumerate(radius) :
            rad  = r_val*kpc #0.5*((radius[indx]+radius[indx-1]) if indx!=0 else 0.5*radius[indx])*kpc
            dr_val = (radius[-1]-radius[-2]) if indx==len(radius)-1 else (radius[indx+1]-radius[indx])*kpc
            
            data = _fourPiNujNu_cloudy (unmod_n[indx], unmod_T[indx], metallicity[indx], redshift, 
                                        indx=indx, cleanup=True if indx==(radius.shape[0]-1) else False) 
            energy = data[:,0]
            luminosityTot += (data[:,1] * (4*np.pi) * dr_val * rad**2)
                                       
        return np.vstack((energy, luminosityTot/energy)).T
    
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    
    for indx, r_val in enumerate(radius) :
        
        r_val  = 0.5*((radius[indx]+radius[indx-1]) if indx!=0 else 0.5*radius[indx])*kpc
        dr_val = ((radius[indx]-radius[indx-1]) if indx!=0 else radius[indx])*kpc
        
        xh   = np.log(Temp/(unmod_T[indx]*np.exp(redisProf.sigH**2/2)))
        PvhT = np.exp(-xh**2/(2*redisProf.sigH**2))/(redisProf.sigH*np.sqrt(2*pi))
        xw   = np.log(Temp/redisProf.TmedVW)
        gvwT = fvw[indx]*np.exp(-xw**2/(2*redisProf.sigW**2))/(redisProf.sigW*np.sqrt(2*pi))
        gvhT = np.piecewise(PvhT, [Temp>=Tcut[indx],], [lambda xp:xp, lambda xp:0.])
        
        #PvwT = interpolate.interp1d(xw, PvwT)
        #PvhT = interpolate.interp1d(xh, PvhT)
        TwM = redisProf.TmedVW*np.exp(-redisProf.sigW**2/2)
        ThM = (unmod_T*((1-fvw)/(1-fmw)))[indx] #modified hot phase
        #Assumtion == Phases are internally isobaric
        
        
        fourPiNujNu_hot  = np.array([fourPiNujNu( (nhot_global[indx]*ThM)/(np.exp(xhp)*(unmod_T[indx]*np.exp(redisProf.sigH**2/2))), 
                                                    np.exp(xhp)*(unmod_T[indx]*np.exp(redisProf.sigH**2/2)), 
                                                    metallicity[indx], redshift)[:,1] \
                                        * gvhT[T_indx]
                                        for T_indx, xhp in enumerate(xh)])
        fourPiNujNu_hot = np.array([ np.trapz(fourPiNujNu_hot[:,E_indx], xh) 
                                        for E_indx in range(EmmSpectrum.energy.shape[0]) ]) # /EmmSpectrum.energy # divide to get photon count
        
        fourPiNujNu_warm = np.array([fourPiNujNu( (nwarm_global[indx]*TwM)/(np.exp(xwp)*redisProf.TmedVW),
                                                      np.exp(xwp)*redisProf.TmedVW,
                                                      metallicity[indx], redshift)[:,1] \
                                        * gvwT[T_indx]
                                        for T_indx, xwp in enumerate(xw)])
        fourPiNujNu_warm = np.array([ np.trapz(fourPiNujNu_warm[:,E_indx], xw) 
                                          for E_indx in range(EmmSpectrum.energy.shape[0]) ]) # /EmmSpectrum.energy # divide to get photon count
        
        # fourPiNujNu_warm = np.zeros_like(fourPiNujNu_hot)
        # erg/s all solid angles covered
        luminosityTot += ( fourPiNujNu_hot + fourPiNujNu_warm ) * (4*np.pi) * dr_val * r_val**2 
        
    return np.vstack((EmmSpectrum.energy, luminosityTot)).T
     