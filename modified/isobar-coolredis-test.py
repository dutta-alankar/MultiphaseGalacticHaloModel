# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:44:07 2022

@author: Alankar
"""

import sys
sys.path.append('..')
import numpy as np
from scipy import interpolate
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isobarcool import IsobarCoolRedistribution
from observable.ColumnDensity import ColumnDensityGen
import matplotlib.pyplot as plt

element = 8

frac = np.loadtxt('ion-frac-Oxygen.txt', skiprows=1, 
                  converters={i+1: lambda x: -np.inf if x==b'--' else x for i in range(element+1)})
ion = 5
fOV = interpolate.interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 
ion = 6
fOVI = interpolate.interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 
ion = 7
fOVII = interpolate.interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 
ion = 8
fOVIII = interpolate.interp1d(frac[:,0], frac[:,ion]) #temperature and ion fraction in log10 

fig = plt.figure()
ax = fig.add_subplot(111)
T = np.logspace(4.2,7.2,400)
plt.loglog(T, 10**fOV(np.log10(T)), label=r'$f_{OV}$' )
plt.loglog(T, 10**fOVI(np.log10(T)), label=r'$f_{OVI}$' )
plt.loglog(T, 10**fOVII(np.log10(T)), label=r'$f_{OVII}$' )
plt.loglog(T, 10**fOVIII(np.log10(T)), label=r'$f_{OVIII}$' )
plt.grid()
plt.ylabel(r'Ionization fraction')
plt.xlabel('Temperature [K]')
plt.legend(loc='best')
ax.yaxis.set_ticks_position('both')
plt.show()

# ____________________________________________________________
# _________________ Isothermal profile _______________________

TmedVH=1.5e6
TmedVW=3.e5
sig = 0.3
cutoff = 10
THotM = TmedVH*np.exp(-sig**2/2)


unmodified = IsothermalUnmodified(THot=THotM,
                      P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                      M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12)
mod_isobar = IsobarCoolRedistribution(unmodified, TmedVH, TmedVW, sig, cutoff)

radius = np.linspace(9.0,250,200) #kpc
nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, _ = mod_isobar.ProfileGen(radius)
MHot, MWarm = mod_isobar.MassGen(radius)
b = np.linspace(9.0,210,240) #kpc
NOVI_PIE  =  ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=6, mode='PIE')
NOVI_CIE  =  ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=6, mode='CIE')
NOVII_PIE = ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=7, mode='PIE')
NOVII_CIE = ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=7, mode='CIE')

MHot = MHot/MSun
MWarm = MWarm/MSun

fvh = 1-fvw
fmh = 1-fmw
nHwarm = (mu/muHp)*nwarm_local

mod_isobar.PlotDistributionGen(radius[100])
mod_isobar.PlotDistributionGen(radius[-50])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, fvw, label='volme')
plt.semilogy(radius, fmw, label='mass')
plt.grid()
plt.ylabel(r'Warm gas filling factor')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
plt.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, nhot_local, label='hot', color='tab:red')
plt.semilogy(radius, nwarm_local, label='warm', color='tab:blue')
plt.semilogy(radius, nhot_global, ls = '--', color='tab:red')
plt.semilogy(radius, nwarm_global, ls = '--', color='tab:blue')
plt.grid()
plt.ylabel(r'$\rm <n_{H}(r)>$ [CGS]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
ax.set_ylim(1e-6,1e-2)
plt.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius[:-1], MWarm, label='warm', color='tab:blue')
plt.semilogy(radius[:-1], MHot, label='Hot', color='tab:red')
plt.grid()
plt.ylabel(r'$\rm M_{gas} [M_{\odot}]$')
plt.xlabel('radius [kpc]')
plt.legend(loc='best')
plt.show()

obsData = np.loadtxt('columndensityData.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVI_CIE, label=r'$N_{OVI}$ (CIE)')
plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVI_PIE, label=r'$N_{OVI}$ (PIE)')
#plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_CIE, label=r'$N_{OVII}$ (CIE)')
#plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_PIE, label=r'$N_{OVII}$ (PIE)')
plt.loglog(obsData[:,0], obsData[:,1], 'o', label=r'Observed $N_{OVI}$')
plt.grid()
plt.ylim(5e13,2e15)
plt.ylabel(r'Column Density [cm^-2]')
plt.xlabel(r'$\rm b/r_{vir}$')
plt.xlim(xmin=5e-2, xmax=1.1)
plt.legend(loc='best')
plt.show()

# ____________________________________________________________
# _________________ Isentropic profile _______________________

TmedVH=1.5e6
TmedVW=3.e5
sig = 0.3
cutoff = 10
THotM = TmedVH*np.exp(-sig**2/2)


unmodified = IsentropicUnmodified()
mod_isobar = IsobarCoolRedistribution(unmodified, TmedVH, TmedVW, sig, cutoff)

radius = np.linspace(9.0,250,200) #kpc
nhot_local, nwarm_local, nhot_global, nwarm_global, fvw, fmw, prs_hot, prs_warm, _ = mod_isobar.ProfileGen(radius)
MHot, MWarm = mod_isobar.MassGen(radius)
b = np.linspace(9.0,210,240) #kpc
NOVI_PIE  =  ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=6, mode='PIE')
NOVI_CIE  =  ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=6, mode='CIE')
NOVII_PIE = ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=7, mode='PIE')
NOVII_CIE = ColumnDensityGen(b, mod_isobar.unmodified.metallicity, mod_isobar, element=8, ion=7, mode='CIE')
MHot = MHot/MSun
MWarm = MWarm/MSun

fvh = 1-fvw
fmh = 1-fmw
nHwarm = (mu/muHp)*nwarm_local

mod_isobar.PlotDistributionGen(radius[100])
mod_isobar.PlotDistributionGen(radius[-50])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, fvw, label='volme')
plt.semilogy(radius, fmw, label='mass')
plt.grid()
plt.ylabel(r'Warm gas filling factor')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
plt.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, nhot_local, label='hot', color='tab:red')
plt.semilogy(radius, nwarm_local, label='warm', color='tab:blue')
plt.semilogy(radius, nhot_global, ls = '--', color='tab:red')
plt.semilogy(radius, nwarm_global, ls = '--', color='tab:blue')
plt.grid()
plt.ylabel(r'$\rm <n_{H}(r)>$ [CGS]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
#ax.set_ylim(1e-6,1e-2)
plt.legend(loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius[:-1], MWarm, label='warm', color='tab:blue')
plt.semilogy(radius[:-1], MHot, label='Hot', color='tab:red')
plt.grid()
plt.ylabel(r'$\rm M_{gas} [M_{\odot}]$')
plt.xlabel('radius [kpc]')
plt.legend(loc='best')
plt.show()

obsData = np.loadtxt('columndensityData.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVI_CIE, label=r'$N_{OVI}$ (CIE)')
plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVI_PIE, label=r'$N_{OVI}$ (PIE)')
#plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_CIE, label=r'$N_{OVII} (CIE)$')
#plt.loglog(b/(mod_isobar.unmodified.Halo.r200*mod_isobar.unmodified.Halo.UNIT_LENGTH/kpc), NOVII_PIE, label=r'$N_{OVII} (PIE)$')
plt.loglog(obsData[:,0], obsData[:,1], 'o', label=r'Observed $N_{OVI}$')
plt.grid()
plt.ylim(5e13,2e15)
plt.ylabel(r'Column Density [cm^-2]')
plt.xlabel(r'$\rm b/r_{vir}$')
plt.xlim(xmin=5e-2, xmax=1.1)
plt.legend(loc='best')
plt.show()
