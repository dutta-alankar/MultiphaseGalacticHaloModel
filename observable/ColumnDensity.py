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
from misc.ionization import interpolate_ionization

def ColumnDensityGen(b_, redisProf, redshift=0.001, element=8, ion=6, mode='CIE'): 
    #takes in b_ in kpc, returns col dens in CGS
    
    metallicity = None
    if isinstance(b_, float) or isinstance(b_, int):
        b_ = np.array([b_])
        #metallicity = np.array([metallicity_])
    #else: 
    #    metallicity = metallicity_
    
    '''
    frac = np.loadtxt('ion-frac-Oxygen.txt', skiprows=1, 
                      converters={i+1: lambda x: -np.inf if x==b'--' else x for i in range(element+1)})
    fOVII = interpolate.interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 
    '''
    ionFrac = interpolate_ionization()
    fIon = ionFrac.interpolate
    
    radius = np.linspace(b_[0]-0.1, redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, 20)    
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    
    #fvh = 1-fvw
    #fmh = 1-fmw
    
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
    Temp  = np.logspace(Tstart, Tstop, 40)
    
    a0 = 4.9e-4 # Asplund et al. 2009
    
    nIon = np.zeros_like(radius)
    for indx, r_val in enumerate(radius) :
        
        xh = np.log(Temp/(unmod_T[indx]*np.exp(redisProf.sigH**2/2)))
        PvhT = np.exp(-xh**2/(2*redisProf.sigH**2))/(redisProf.sigH*np.sqrt(2*pi))
        xw = np.log(Temp/redisProf.TmedVW)
        gvwT = fvw[indx]*np.exp(-xw**2/(2*redisProf.sigW**2))/(redisProf.sigW*np.sqrt(2*pi))
        gvhT = np.piecewise(PvhT, [Temp>=Tcut[indx],], [lambda xp:xp, lambda xp:0.])
        
        #PvwT = interpolate.interp1d(xw, PvwT)
        #PvhT = interpolate.interp1d(xh, PvhT)
        TwM = redisProf.TmedVW*np.exp(-redisProf.sigW**2/2)
        ThM = (unmod_T*((1-fvw)/(1-fmw)))[indx] #modified hot phase
        #Assumtion == Phases are internally isobaric
        
        # print('Ionization interpolation')
        fIonHot  = 10**np.array([fIon( (nhot_global[indx]*ThM)/(np.exp(xhp)*(unmod_T[indx]*np.exp(redisProf.sigH**2/2))), 
                                             np.exp(xhp)*(unmod_T[indx]*np.exp(redisProf.sigH**2/2)), 
                                             metallicity[indx], redshift, 
                                             element, ion, mode) for xhp in xh])
        # print(fIonHot)
        fIonWarm =  10**np.array([fIon( (nwarm_global[indx]*TwM)/(np.exp(xwp)*redisProf.TmedVW),
                                             np.exp(xwp)*redisProf.TmedVW,
                                             metallicity[indx], redshift,
                                             element, ion, mode) for xwp in xw])
        # print(fIonWarm)
        # if indx == 0 or indx==10:
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure(figsize=(13,10))
        #     ax  =  plt.gca()
        #     plt.loglog(Temp, fIonHot,  label='interpolated Hot', linewidth=5)
        #     plt.loglog(Temp, fIonWarm, label='interpolated Warm', linewidth=5)
        #     ax.grid()   
        #     ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
        #     ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
        #     ax.set_ylabel(r'Ionization fraction', size=28, color='black') 
        #     ax.set_xlabel(r'Temperature [K]', size=28, color='black')
        #     plt.xlim(xmin=10**4.4, xmax=10**7.6)
        #     plt.ylim(ymin=1e-8, ymax=1.1)
        #     lgnd = ax.legend(loc='best', framealpha=0.3, prop={'size': 20}, title_fontsize=24) #, , , bbox_to_anchor=(0.88, 0))
        #     ax.set_title(f'Element=%d Ion=%d Mode=%s'%(element, ion, mode), fontsize=26)
        #     plt.show()
        
        # print('Integration')
        hotInt  = np.trapz( (gvhT/(np.exp(xh)*(unmod_T[indx]*np.exp(redisProf.sigH**2/2))))*fIonHot, xh)
        
        warmInt = np.trapz( (gvwT/((np.exp(xw)*redisProf.TmedVW)))*fIonWarm, xw)
        #P(T) dT = g(x w_h) dx w_h
        
        nIon[indx] = a0*metallicity[indx]*(mu[indx]/muHp[indx])*(
                     (nhot_global[indx]*ThM)* hotInt + (nwarm_global[indx]*TwM)* warmInt)
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(13,10))
    # ax  =  plt.gca()
    # plt.semilogy(radius, nhot_global,  label='Hot', linewidth=5)
    # plt.semilogy(radius, nwarm_global, label='Warm', linewidth=5)
    # plt.semilogy(radius, nIon, label='Ion e=%d i=%d'%(element,ion), linewidth=5)
    # ax.grid()   
    # ax.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
    # ax.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
    # ax.set_ylabel(r'Ionization fraction', size=28, color='black') 
    # ax.set_xlabel(r'Temperature [K]', size=28, color='black')
    # lgnd = ax.legend(loc='best', framealpha=0.3, prop={'size': 20}, title_fontsize=24) #, , , bbox_to_anchor=(0.88, 0))
    # ax.set_title(f'Element=%d Ion=%d Mode=%s'%(element, ion, mode), fontsize=26)
    # # plt.xlim(xmin=10**4.4, xmax=10**7.6)
    # # plt.ylim(ymin=1e-8, ymax=1.1)
    # plt.show()
    
    nIon = interpolate.interp1d(radius, nIon, fill_value="extrapolate") #CGS
    # print('Ion calculation complete!')
    coldens = np.zeros_like(b_)
    
    epsilon = 1e-6
    for indx, b_val in enumerate(b_):
        coldens[indx] = 2*integrate.quad(lambda r: nIon(r)*r/np.sqrt(r**2-b_val**2), 
                                b_val*(1+epsilon), redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc)[0] #kpc cm-3
    
    if len(b_) == 1: return coldens[0]*kpc
    else: return coldens*kpc
