#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:42:10 2023

@author: alankar
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# from scipy.integrate import simpson
sys.path.append('..')
from misc.ionization import interpolate_ionization
from misc.HaloModel import HaloModel
from misc.ColDensCOSData import observedColDens
from misc.constants import *

N_pdf = lambda x, mu, sig: (1./(np.sqrt(2*np.pi)*sig))*np.exp(-(x-mu)*(x-mu)/(2.*sig*sig))

def abundance(element):
    file_path = os.path.realpath(__file__)
    dir_loc   = os.path.split(file_path)[:-1]
    abn_file  = os.path.join(*dir_loc,'..','misc','cloudy-data', 'solar_GASS10.abn')
    _tmp = None
    with open(abn_file, 'r') as file:
        _tmp = file.readlines()
        
    abn = np.array([ float(element.split()[-1]) for element in _tmp[2:32] ]) #till Zinc
    a0  = abn[element-1]
    return a0

X_solar = 0.7154
Y_solar = 0.2703
Z_solar = 0.0143

Xp = lambda metallicity: X_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
Yp = lambda metallicity: Y_solar*(1-metallicity*Z_solar)/(X_solar+Y_solar)
Zp = lambda metallicity: metallicity*Z_solar # Z varied independent of nH and nHe
        
Z0   = 1.0
ZrCGM = 0.3
p = Z0/ZrCGM
metallicity = 1.5*(p-(p**2-1)*np.arcsin(1./np.sqrt(p**2-1)))*Z0*np.sqrt(p**2-1)
redshift = 0.001
nH = 1.e-3
M200 = 1e13 #MSun
halo = HaloModel(M200=M200)
r200 = halo.r200*(halo.__class__.UNIT_LENGTH/kpc) #kpc
PbykB = 1e2

f_Vh, f_Vw, f_Vc, x_h, x_w, x_c, sig_h, sig_w, sig_c, T_u = np.load('parameters.npy')

phases_data = np.load('3PhasePdf-LogTu=%.1fK.npy'%np.log10(T_u))
Temperature = 10.**phases_data[:,0]
x = np.log(Temperature/T_u)

V_pdf_fv = f_Vh**2*N_pdf(x,x_h,sig_h) + f_Vw**2*N_pdf(x,x_w,sig_w) + f_Vc**2*N_pdf(x,x_c,sig_c)

def nIon_avg(element, ion, mode):
    ionFrac = interpolate_ionization()
    fIon = ionFrac.interpolate
    fIon = 10.**np.array([fIon(PbykB/T_val, T_val, metallicity, redshift, element, ion, mode)   for T_val in Temperature])
    nIon_avg = np.trapz(abundance(element)*metallicity*fIon*(mp/mH)*Xp(metallicity)*nH*V_pdf_fv/Temperature, Temperature)
    return nIon_avg

ionization = interpolate_ionization()
num_dens = ionization.num_dens

mode = 'PIE'
   
ne =   np.array([num_dens(PbykB/T_val, T_val, metallicity, redshift, mode, 'electron') for T_val in Temperature])
ni =   np.array([num_dens(PbykB/T_val, T_val, metallicity, redshift, mode, 'ion')      for T_val in Temperature])

# print('ne ', ne)
# print('ni ', ni)

ne_avg   = np.trapz(ne*V_pdf_fv/Temperature, Temperature)
ni_avg   = np.trapz(ni*V_pdf_fv/Temperature, Temperature)

# print('ne_avg ', ne_avg)
# print('ni_avg ', ni_avg)

b = np.linspace(9.0,1.1*r200,200) #kpc
column_length = 2*np.sqrt(r200**2-b**2)

def IonColumn(element, ion, name, ylim=None, fignum=None, color=None):
    fig = None
    if (fignum == None): 
        plt.figure(figsize=(13,10))
    else:
        fig = plt.figure(num=fignum, figsize=(13,10))
    if (color == None): color='tab:blue'
    
    NIon = np.nan_to_num(nIon_avg(element, ion, mode)*column_length*kpc) # cm^-2
    plt.plot(b/r200, NIon, color=color, label=r'$\rm N_{%s, %s}$'%(name, mode), linewidth=5)
      
    gal_id_min, gal_id_max, gal_id_detect, \
    rvir_select_min, rvir_select_max, rvir_select_detect,\
    impact_select_min, impact_select_max, impact_select_detect,\
    coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element=name)
    
    yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
    plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
                 fmt='o', color=color, label=r'$\rm N_{%s, obs}$'%name, markersize=12)
    plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color=color, markersize=12)
    plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color=color, markersize=12)
    
    plt.xscale('log')
    plt.yscale('log')
    if(ylim!=None): plt.ylim(*ylim)
    plt.xlim(6e-2, 1.2)
    plt.xlabel(r'$b/R_{vir}$', size=28)
    plt.ylabel(r'Column density $[cm^{-2}]$' ,size=28)
    
    plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
    plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
    plt.grid()
    plt.tight_layout()
    # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    # leg.set_title("Column density predicted by three phase model",prop={'size':20})
    if (fignum == None): 
        leg = plt.legend(loc='upper right', ncol=1, fancybox=True, fontsize=25)
        plt.savefig('./N_%s-3p.png'%name, transparent=True)
        # plt.show()
        plt.close()

# --------------------- DM and EM ------------------
DM = ne_avg*column_length*1e3 #cm^-3 pc
EM = ne_avg*ni_avg*column_length*1e3 #cm^-6 pc

# DM
plt.figure(figsize=(13,10))
plt.plot(b/r200, DM, color='firebrick', linewidth=5)
plt.xscale('log')
plt.yscale('log')
plt.ylim(4.0, 60.)
plt.xlim(6e-2, 1.2)
plt.xlabel(r'$b/R_{vir}$', size=28)
plt.ylabel(r'DM [$cm^{-3} pc$]' ,size=28)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
# set the linewidth of each legend object
# for legobj in leg.legendHandles:
# leg.set_title("Column density predicted by three phase model",prop={'size':20})
plt.savefig('./DM-3p.png', transparent=True)
# plt.show()
plt.close()

# EM
plt.figure(figsize=(13,10))
plt.plot(b/r200, EM*1e3, color='firebrick', linewidth=5)
plt.xscale('log')
plt.yscale('log')
plt.ylim(2e-2, 0.3)
plt.xlim(6e-2, 1.2)
plt.xlabel(r'$b/R_{vir}$', size=28)
plt.ylabel(r'EM [$\times 10^{-3} cm^{-6} pc$]' ,size=28)
# leg = plt.legend(loc='lower left', ncol=1, fancybox=True, fontsize=25)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
# set the linewidth of each legend object
# for legobj in leg.legendHandles:
# leg.set_title("Column density predicted by three phase model",prop={'size':20})
plt.savefig('./EM-3p.png', transparent=True)
# plt.show()
plt.close()

# ---------- Individual Ions -------------------------
element = 8
ion = 6
name = 'O VI'
IonColumn(element, ion, name, ylim=(10**14.0, 10.**15.3))

element = 12
ion = 2
name = 'Mg II'
IonColumn(element, ion, name)

element = 14
ion = 4
name = 'Si IV'
IonColumn(element, ion, name)

element = 16
ion = 3
name = 'S III'
IonColumn(element, ion, name)

element = 7
ion = 5
name = 'N V'
IonColumn(element, ion, name)

element = 6
ion = 3
name = 'C III'
IonColumn(element, ion, name)

element = 6
ion = 2
name = 'C II'
IonColumn(element, ion, name)

# all together
fignum = 100
fig = plt.figure(figsize=(13,10), num=fignum)
# print('Figures: ', plt.get_fignums())
element = 8
ion = 6
name = 'O VI'
IonColumn(element, ion, name, fignum=fignum, color='coral')

element = 12
ion = 2
name = 'Mg II'
IonColumn(element, ion, name, fignum=fignum, color='lightseagreen')

fig = plt.figure(num=fignum)
leg = plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize=25)
plt.ylim(2e10, 2e16)
plt.grid()
plt.savefig('N_OVI+MgII-3p.png', transparent=True)
plt.show()
# plt.close()
