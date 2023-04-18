#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:27:44 2023

@author: alankar
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
import os


## Plot Styling
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['lines.linewidth'] = 3.0
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.minor.width'] = 0.45
matplotlib.rcParams['xtick.minor.width'] = 0.45
matplotlib.rcParams['ytick.major.size'] = 4.0
matplotlib.rcParams['xtick.major.size'] = 4.0
matplotlib.rcParams['ytick.minor.size'] = 2.0
matplotlib.rcParams['xtick.minor.size'] = 2.0
matplotlib.rcParams['legend.handlelength'] = 2
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams['axes.axisbelow'] = True
npoints = 100

# os.system('mkdir -p ./isoth')
TmedVH=1.5e6
TmedVW=3.e5
sig = 0.3
THotM = TmedVH*np.exp(-sig**2/2)

# PIE Unmidified Isothermal
unmodified = IsothermalUnmodified(THot=THotM,
                      P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                      M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                      redshift=0.001, ionization='PIE')
    
def generate_clouds(fvc, fvw, profile):
    r0 = profile.Halo.r0*profile.Halo.UNIT_LENGTH/kpc # kpc
    r_vir = profile.Halo.r200*profile.Halo.UNIT_LENGTH/kpc # kpc
    r_cold = 1 # pc
    r_warm = 10*rc_cold # pc
    
    N_cold = fvc*(r_vir*1e3/r_cold)**3
    N_warm = fvc*(r_vir*1e3/r_warm)**3
    
    # npoints = 1000
    # rho, prsTh, prsnTh, prsTurb, prsTot, nH, mu = profile.ProfileGen(
    #     np.linspace(profile.Halo.r0*profile.Halo.UNIT_LENGTH/kpc,
    #                                r_vir, npoints))
    r_till = r_vir*1e3 - r_cold
    pos_cold = np.random.uniform(r0, r_till, N_cold)
    r_till = r_vir*1e3 - r_warm
    pos_warm = np.random.uniform(r0, r_till, N_cold)
     
    return (pos_cold, pos_warm)

def shoot_lines(b, r_vir, point, nlines=20, res=100):
    if (b>=r_vir): print ('Problem!')   
    los = np.zeros((nlines, res, 3), dtype=np.float64)
    phi = np.linspace(0, 2*np.pi, nlines)
    sin_theta = np.zeros_like(phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    for i, _phi in enumerate(phi):
        _tmp = point[0]*np.cos(_phi) + point[1]*np.sin(_phi)
        A = _tmp**2 + point[2]**2
        B = -2*b*_tmp
        C = b**2 - point[2]**2
        Dis = np.sqrt(B**2-4*A*C)
        sin_theta[i] = max ( (-B + Dis)/(2*A), (-B - Dis)/(2*A) )
    cos_theta = np.sqrt(1-sin_theta**2)
    
    distance = np.linspace(0, 2*np.sqrt(r_vir**2-b**2), res)
    
    for line in range(nlines):
        direction = [-point[0]+b*sin_theta[line]*cos_phi[line],
                     -point[1]+b*sin_theta[line]*sin_phi[line],
                     -point[2]+b*cos_theta[line]]
        direction = np.array(direction)/np.sqrt( np.sum(np.array(direction)**2) )
        
        along_los = np.array([distance*direction[0], 
                              distance*direction[1], 
                              distance*direction[2]])
        los[line,:,0] = point[0] + along_los[0,:]
        los[line,:,1] = point[1] + along_los[1,:]
        los[line,:,2] = point[2] + along_los[2,:]
    return los

npoints = 5
r_vir = 220 # kpc
b  = 180 # kpc
theta = np.random.uniform(0, np.pi, npoints)
phi   = np.random.uniform(0, 2*np.pi, npoints)

x, y, z = r_vir*np.sin(theta)*np.cos(phi), r_vir*np.sin(theta)*np.sin(phi), r_vir*np.cos(theta)

for i in range(npoints):
    point = np.array([ x[i], y[i], z[i] ])
    los = shoot_lines(b, r_vir, point)

    for line in range(los.shape[0]):
        plt.plot(los[line,:,0], los[line,:,1])

plt.set_aspect('equal')    
plt.show()    