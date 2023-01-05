# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:58:32 2022

@author: Alankar
"""
import sys
sys.path.append('..')
from misc.constants import *
import isent
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

Npts = 200
Model = isent.IsentropicUnmodified()
radius = np.linspace(Model.Halo.r0*Model.Halo.UNIT_LENGTH/kpc, 
                     Model.rCGM*Model.UNIT_LENGTH/kpc, Npts) #kpc
rho, PTh, PNTh, PTurb, Ptot, nH, mu = Model.ProfileGen(radius)

n  = Model.ndens
T  = Model.Temperature
Mcor = integrate.cumtrapz(4*pi*(radius*kpc)**2*rho, radius*kpc)/MSun
#Mcor = rho*(4/3)*pi*radius**3*Model.UNIT_MASS/MSun
met  = Model.metallicity
ZmetAvg = integrate.cumtrapz(4*pi*(radius*kpc)**2*rho*met, radius*kpc)/(Mcor*MSun)
Mmet    = Model.fZ*Mcor*ZmetAvg #MSun

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, nH)
plt.semilogy(radius, 1.3e-5*(radius/Model.rCGM)**(-0.93)*cm**(-3), ls=':', label='Fit')
plt.grid()
plt.legend(loc='best')
plt.ylabel(r'$\rm n_{H}$ [CGS]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
#ax.set_xlim(0,250)
ax.set_ylim(5e-6,1e-3)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.log10(T), mu)
plt.grid()
plt.ylabel(r'$\mu$ ($n_H$)')
plt.xlabel(r'Temperature [$K$]')
ax.yaxis.set_ticks_position('both')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, T)
plt.semilogy(radius, 2.7e5*(radius/Model.rCGM)**(-0.62)*K, ls=':', label='Fit')
plt.grid()
plt.legend(loc='best')
plt.ylabel(r'Temperature [K]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
#ax.set_xlim(0,250)
ax.set_ylim(8e4,4.5e6)
plt.show()
  
fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, PTh/kB, label='Thermal')
plt.semilogy(radius, Ptot/kB, label='Total')
plt.semilogy(radius, 22.1*(radius/Model.rCGM)**(-1.35)*K*cm**(-3), ls=':', label='Fit')
plt.grid()
plt.legend(loc='best')
plt.ylabel(r'$\rm P/k_{B}$ [CGS]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
#ax.set_xlim(0,250)
ax.set_ylim(1,1e4)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(radius, PTh/Ptot, label='Thermal')
plt.plot(radius, PNTh/Ptot, label='CR or B')
plt.plot(radius, PTurb/Ptot, label='Turb')
plt.grid()
plt.legend(loc='best')
plt.ylabel(r'$\rm P_{i}/P_{tot}$')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
#ax.set_xlim(0,250)
ax.set_ylim(0,1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(radius, np.sqrt(4*pi*PNTh)*1e9)
plt.plot([Model.Halo.r0*Model.Halo.UNIT_LENGTH/kpc, Model.rCGM*Model.UNIT_LENGTH/kpc], 
         [940, 110],'o', color='black')
plt.grid()
plt.title(r'$B=\sqrt{4 \pi P_{nt}}$')
plt.ylabel(r'Magnetic field [nG]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius[:-1], Mcor, label='gas')
plt.semilogy(radius[:-1], Mmet, label='metal')
plt.vlines(Model.Halo.r200*Model.Halo.UNIT_LENGTH/kpc, 
           1e2, 1e18, colors='black', linestyles='--', label=r'$\rm r_{vir}$')
plt.grid()
plt.legend(loc='best')
plt.title('Coronal mass')
plt.ylabel(r'$\rm M(r)\ [M_{\odot}]$')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_ylim(1e4,2e11)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(radius, met)
plt.vlines(Model.rZ*Model.UNIT_LENGTH/kpc, 
           0, 2, colors='black', linestyles='--', label=r'$\rm r_{Z}$')
plt.grid()
plt.legend(loc='best')
plt.title('Metallicity')
plt.ylabel(r'$\rm Z(r)\ [Z_{\odot}]$')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_ylim(0.2,1.2)
plt.show()