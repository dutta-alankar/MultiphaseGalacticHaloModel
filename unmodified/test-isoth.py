# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:53:17 2022

@author: Alankar
"""

import sys
sys.path.append('..')
from misc.constants import *
import isoth
import numpy as np
from scipy import integrate
from scipy import interpolate 
import matplotlib.pyplot as plt

sig = 0.3
Temperature = 1.5e6*np.exp(-sig**2/2)

Npts = 200
Model = isoth.IsothermalUnmodified(THot=Temperature)
radius = np.linspace(9.0, 250, Npts) #kpc
rho, prsTh, _, prsTurb, prsTot = Model.ProfileGen(radius)


n = rho/(mu*mp)
nHhot  = (mu/muHp)*n #CGS

isobaric = 0
#LAMBDA = np.loadtxt('cooltable.dat')
LAMBDA = 10**np.loadtxt('../modified/hazy_coolingcurve_0.5.txt', skiprows=1)
LAMBDA = interpolate.interp1d(LAMBDA[:,0], LAMBDA[:,1])
tcool = lambda ndens, Temp:(4.34*(1.5+isobaric)*kB*Temp/(ndens*LAMBDA(Temp)))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(radius, tcool(n, Temperature)/(1e9*yr))
plt.grid()
plt.ylabel(r'tcool, hot [Gyr]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, prsTot/kB)
plt.semilogy([Model.Halo.r0*Model.Halo.UNIT_LENGTH/kpc, 250,], [4580, 230,], 'o', color = 'black')
plt.grid()
plt.ylabel(r'$\rm <P_{hot}(r)>/k_B$ [CGS]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
ax.set_ylim(50,1e4)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogy(radius, nHhot)
plt.semilogy([50, 100], [0.83e-4, 1.3e-4], 'o', color = 'black')
plt.grid()
plt.ylabel(r'$\rm <n_{H,hot}(r)>$ [CGS]')
plt.xlabel('radius [kpc]')
ax.yaxis.set_ticks_position('both')
ax.set_xlim(0,250)
ax.set_ylim(1e-6,1e-2)
plt.show()