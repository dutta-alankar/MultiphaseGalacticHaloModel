#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:56:00 2022

@author: alankar
"""

import sys
import numpy as np
import os
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson
from scipy.optimize import root
sys.path.append('..')
from misc.constants import *
from misc.ionization import interpolate_ionization

def ColumnDensityGen(b_, redisProf): 
    #takes in b_ in kpc, returns col dens in CGS
    mode = redisProf.ionization
    redshift = redisProf.redshift
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
    ionization = interpolate_ionization()
    num_dens = ionization.num_dens
    
    X_solar = 0.7154
    Y_solar = 0.2703
    Z_solar = 0.0143

    Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
    Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
    Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
    # mu for 100% ionization and Solar for quick t_cool calculation as mu doesn't change a lot for ranges of temperature and density of interest
    #mu = 1./(2*Xp+(3/4)*Yp+(9/16)*Zp)
    
    
    radius = np.linspace(b_[0]-0.1, redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc+0.1, 20)    
    nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, Tcut = redisProf.ProfileGen(radius)
    metallicity = redisProf.unmodified.metallicity
    
    Temp  = redisProf.TempDist
    
    THotM  = (redisProf.prs_hot/redisProf.rhohot_local)*(kB/(redisProf.mu_hot*mp))
    TWarmM = (redisProf.prs_warm/redisProf.rhowarm_local)*(kB/(redisProf.mu_warm*mp))
    
    ne = np.zeros_like(radius)
    for indx, r_val in enumerate(radius) :    
        xh = np.log(Temp/(THotM[indx]*np.exp(redisProf.sigH**2/2)))
        PvhT = np.exp(-xh**2/(2*redisProf.sigH**2))/(redisProf.sigH*np.sqrt(2*pi))
        xw = np.log(Temp/redisProf.TmedVW)
        gvwT = fvw[indx]*np.exp(-xw**2/(2*redisProf.sigW**2))/(redisProf.sigW*np.sqrt(2*pi))
        gvhT = np.piecewise(PvhT, [Temp>=Tcut[indx],], [lambda xp:xp, lambda xp:0.])
        
        #PvwT = interpolate.interp1d(xw, PvwT)
        #PvhT = interpolate.interp1d(xh, PvhT)
        # TwM = redisProf.TmedVW*np.exp(-redisProf.sigW**2/2)
        # ThM = (ThotM*((1-fvw)/(1-fmw)))[indx] #modified hot phase
        #Assumtion == Phases are internally isobaric
        
        ndensHot  = redisProf.prs_hot[indx]/(kB*Temp) #CGS
        ndensWarm = redisProf.prs_warm[indx]/(kB*Temp) #CGS
        
        nHhot  = redisProf.nHhot_local[indx]*redisProf.TmedVH[indx]*np.exp(-redisProf.sigH**2/2)/Temp #CGS
        nHwarm = redisProf.nHwarm_local[indx]*redisProf.TmedVW*np.exp(-redisProf.sigW**2/2)/Temp #CGS
        # 10**np.array([root(lambda LognH: 10**LognH - 
        #                     ndensHot[indx]*Xp(metallicity[indx])*
        #                     (interpolate_ionization().mu(10**LognH, Temp[i] , metallicity[indx], redshift, mode)), -3).x[0] for i in range(Temp)])
                               
        # print('Ionization interpolation')
        neHot  = np.array([num_dens(      nHhot[i], 
                                             Temp[i], 
                                             metallicity[indx], redshift, 
                                             mode=mode, part_type='electron' ) for i,xhp in enumerate(xh)])
        
        # 10**np.array([root(lambda LognH: 10**LognH - 
        #                     ndensWarm[indx]*Xp(metallicity[indx])*
        #                     (interpolate_ionization().mu(10**LognH, TwM , metallicity[indx], redshift, mode)), -3).x[0] for i in range(Temp)])
        
        # print(fIonHot)
        neWarm =  np.array([fIon(     nHwarm[i],
                                             Temp[i],
                                             metallicity[indx], redshift,
                                             mode=mode, part_type='electron' ) for i,xwp in enumerate(xw)])
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
        # print("Test: ", redisProf.prs_hot[5], redisProf.prs_warm[5])
        hotInt  = (1-fvw[indx])*simpson( (neHot*gvhT, xh) #global density sensitive
        
        warmInt = fvw[indx]*simpson( (neWarm*gvwT, xw)
        #P(T) dT = g(x w_h) dx w_h
        
        ne[indx] = a0*metallicity[indx]*(hotInt + warmInt)
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(13,10))
    # ax  =  plt.gca()
    # plt.semilogy(radius, nhot_global,  label='Hot', linewidth=5)
    # plt.semilogy(radius, nwarm_global, label='Warm', linewidth=5)
    # plt.semilogy(radius, ne, label='Ion e=%d i=%d'%(element,ion), linewidth=5)
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
    
    ne = interpolate.interp1d(radius, ne, fill_value="extrapolate") #CGS
    # print('Ion calculation complete!')
    coldens = np.zeros_like(b_)
    
    epsilon = 1e-6
    for indx, b_val in enumerate(b_):
        coldens[indx] = 2*integrate.quad(lambda r: ne(r)*r/np.sqrt(r**2-b_val**2), 
                                b_val*(1+epsilon), redisProf.unmodified.rCGM*redisProf.unmodified.UNIT_LENGTH/kpc)[0] #kpc cm-3
    
    if len(b_) == 1: return coldens[0]*kpc
    else: return coldens*kpc

