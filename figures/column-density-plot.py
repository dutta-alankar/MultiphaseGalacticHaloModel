#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:42:48 2022

@author: alankar
"""
import numpy as np


class observedColDens:
    
    def __init__(self, galaxyDataFile='apjs456058t3_mrt.txt', galaxySizeFile='VirialRad.txt'):
        self.galaxyDataFile = galaxyDataFile
        self.galaxySizeFile = galaxySizeFile
        
    def readData(self, filename):
        lines = None
        start = 41
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        galaxies = []
        names = []
        impact = []
        limit = []
        coldens = []
        e_coldens = []
        for line in lines[start:]:
            # print(line)
            galaxies.append(line[:17].strip())
            names.append(line[29:35])
            impact.append(float(line[25:28]))
            if line[78]==' ':
                limit.append('e')
            elif line[78]=='<':
                limit.append('l')
            elif line[78]=='>':
                limit.append('g')
            if (line[80:85] != '     '):
                coldens.append(float(line[80:85]))
                if limit[-1]=='e':
                    e_coldens.append(float(line[86:90]))
                else:
                    e_coldens.append(0.)
            else:
                if line[62]==' ':
                    limit[-1] = 'e'
                elif line[62]=='<':
                    limit[-1] = 'l'
                elif line[62]=='>':
                    limit[-1] = 'g'
                coldens.append(float(line[64:69]))
                if limit[-1]=='e':
                    e_coldens.append(float(line[70:74]))
                else:
                    e_coldens.append(0.)
        
        self.galaxies  = galaxies
        self.names     = names
        self.impact    = impact
        self.limit     = limit
        self.coldens   = coldens
        self.e_coldens = e_coldens
    
    def RreadRvir(self, filename):
        lines = None
        start = 6
        stop = 50
        with open(filename, 'r') as file:
            lines = file.readlines()
       
        qsoGal_names = []
        rvir = []
        for line in lines[start:stop]:
             # print(line) 
             values = line.split()
             qsoGal_names.append((values[0]+'_'+values[1]).strip())
             rvir.append(float(values[-1]))
        
        self.qsoGal_names =  qsoGal_names
        self.rvir         = rvir
    
    def col_densGen(self, element = 'O VI'):
        self.readData(self.galaxyDataFile)
        self.RreadRvir(self.galaxySizeFile)
        
        indices = [index for index in range(len(self.names)) if self.names[index].strip()==element ]
        coldens_min = []
        coldens_max = []
        coldens_detect = []
        e_coldens_detect = []
        gal_id_min = []
        gal_id_max = []
        gal_id_detect = []
        impact_select_min = []
        rvir_select_min = []
        impact_select_max = []
        rvir_select_max = []
        impact_select_detect = []
        rvir_select_detect = []
        
        for indx in indices:
            if   (self.limit[indx]=='l'): 
                gal_id_max.append(self.galaxies[indx])
                coldens_max.append(self.coldens[indx])
                impact_select_max.append(self.impact[indx])
                rvir_select_max.append(self.rvir[self.qsoGal_names.index(self.galaxies[indx])])
            elif (self.limit[indx]=='g'): 
                gal_id_min.append(self.galaxies[indx])
                coldens_min.append(self.coldens[indx])
                impact_select_min.append(self.impact[indx])
                rvir_select_min.append(self.rvir[self.qsoGal_names.index(self.galaxies[indx])])
            elif (self.limit[indx]=='e'): 
                gal_id_detect.append(self.galaxies[indx])
                coldens_detect.append(self.coldens[indx])
                e_coldens_detect.append(self.e_coldens[indx])
                impact_select_detect.append(self.impact[indx])
                rvir_select_detect.append(self.rvir[self.qsoGal_names.index(self.galaxies[indx])])
                
        impact_select_min = np.array(impact_select_min)
        rvir_select_min = np.array(rvir_select_min)
        impact_select_max = np.array(impact_select_max)
        rvir_select_max = np.array(rvir_select_max)
        impact_select_detect = np.array(impact_select_detect)
        rvir_select_detect = np.array(rvir_select_detect)
        coldens_min = np.array(coldens_min)
        coldens_max = np.array(coldens_max)
        coldens_detect = np.array(coldens_detect)
        e_coldens_detect = np.array(e_coldens_detect)
        
        return (gal_id_min, gal_id_max, gal_id_detect, 
                rvir_select_min, rvir_select_max, rvir_select_detect,
                impact_select_min, impact_select_max, impact_select_detect,
                coldens_min, coldens_max, coldens_detect, e_coldens_detect)
    
        
import matplotlib.pyplot as plt
observation = observedColDens()

element = 'O VI'
gal_id_min, gal_id_max, gal_id_detect, \
rvir_select_min, rvir_select_max, rvir_select_detect,\
impact_select_min, impact_select_max, impact_select_detect,\
coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element = element)

plt.figure(figsize=(13,10))
yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
             fmt='o', color='tab:blue', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color='tab:blue', markersize=12)
plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color='tab:blue', markersize=12)

#model
R_vir = 211.94 #kpc

NOVI_PIE = np.load('./isoth/NOVI_PIE.npy')
NOVI_CIE = np.load('./isoth/NOVI_CIE.npy')

plt.plot(NOVI_PIE[:,0]/R_vir, np.nan_to_num(NOVI_PIE[:,1]), color='tab:blue', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NOVI_CIE[:,0]/R_vir, np.nan_to_num(NOVI_CIE[:,1]), color='tab:blue', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

element = 'N V'
gal_id_min, gal_id_max, gal_id_detect, \
rvir_select_min, rvir_select_max, rvir_select_detect,\
impact_select_min, impact_select_max, impact_select_detect,\
coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element = element)

yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color='tab:olive', markersize=12)
plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color='tab:olive', markersize=12)

#model
NNV_PIE = np.load('./isoth/NNV_PIE.npy')
NNV_CIE = np.load('./isoth/NNV_CIE.npy')

plt.plot(NNV_PIE[:,0]/R_vir, np.nan_to_num(NNV_PIE[:,1]), color='tab:olive', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NNV_CIE[:,0]/R_vir, np.nan_to_num(NNV_CIE[:,1]), color='tab:olive', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

plt.xscale('log')
plt.yscale('log')
plt.ylim(10.**11.8, 10.**15.3)
plt.xlim(6e-2, 1.2)
plt.xlabel(r'$b/R_{vir}$', size=28)
plt.ylabel(r'Column density $[cm^{-2}]$' ,size=28)
leg = plt.legend(loc='lower left', ncol=3, fancybox=True, fontsize=25)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
leg.set_title("Isothermal profile with isochoric redistribution", prop={'size':20})
plt.savefig('./isoth/ColumnDensity-warm.png', transparent=True)
plt.show()
plt.close()

plt.figure(figsize=(13,10))
'''
Reference: Miller Bregman 2013
Number	Name	l	b	EW	Error	$N_{{\rm O}\,{\scriptsize{VII}},{\rm thin}}$	Error	N_model, thin	$N_{{\rm O}\,{\scriptsize{VII}},{\rm saturated}}$	Error	N_model, saturated	
(deg)	(deg)	(mAring)	(mAring)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	
1	Mrk 421	179.83	65.03	11.8	0.8	4.12	2.53	4.62	5.36	3.61	4.54	
2	PKS 2155-304	17.73	-52.24	13.7	1.9	4.79	2.60	7.55	6.56	4.06	8.97	
'''

#model
element = 'O VII'
NOVII_PIE = np.load('./isoth/NOVII_PIE.npy')
NOVII_CIE = np.load('./isoth/NOVII_CIE.npy')

plt.plot(NOVII_PIE[:,0]/R_vir, np.nan_to_num(NOVII_PIE[:,1]), color='tab:blue', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NOVII_CIE[:,0]/R_vir, np.nan_to_num(NOVII_CIE[:,1]), color='tab:blue', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

#obs
NOVII_obs = 15.68 
NOVII_err = 0.27
yerr = np.log(10)*NOVII_err*10.**NOVII_obs
#plt.errorbar(np.array([6.5e-2,]), 2*10.**np.array([NOVII_obs,]), yerr=2*np.array([yerr,]), uplims=True,
#             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.axhspan(2*10.**NOVII_obs, 2*(10.**NOVII_obs-yerr), 
                 color='tab:blue', alpha=0.2, zorder=0)

element = 'O VIII'
NOVIII_PIE = np.load('./isoth/NOVIII_PIE.npy')
NOVIII_CIE = np.load('./isoth/NOVIII_CIE.npy')

plt.plot(NOVIII_PIE[:,0]/R_vir, np.nan_to_num(NOVIII_PIE[:,1]), color='tab:olive', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NOVIII_CIE[:,0]/R_vir, np.nan_to_num(NOVIII_CIE[:,1]), color='tab:olive', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

#obs
NOVIII_obs = NOVII_obs-np.log10(4) 
NOVIII_err = NOVII_err-np.log10(4)
yerr = np.log(10)*NOVIII_err*10.**NOVIII_obs
#plt.errorbar(np.array([6.5e-2,]), 2*10.**np.array([NOVII_obs,]), yerr=2*np.array([yerr,]), uplims=True,
#             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.axhspan(2*10.**NOVIII_obs, 2*(10.**NOVIII_obs-yerr), 
                 color='tab:olive', alpha=0.2, zorder=0)

plt.xscale('log')
plt.yscale('log')
plt.ylim(10**14.0, 10.**16.1)
plt.xlim(6e-2, 1.2)
plt.xlabel(r'$b/R_{vir}$', size=28)
plt.ylabel(r'Column density $[cm^{-2}]$' ,size=28)
leg = plt.legend(loc='lower left', ncol=2, fancybox=True, fontsize=25)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
leg.set_title("Isothermal profile with isochoric redistribution",prop={'size':20})
plt.savefig('./isoth/ColumnDensity-hot.png', transparent=True)
plt.show()

'''
element = 'Si IV'
gal_id_min, gal_id_max, gal_id_detect, \
rvir_select_min, rvir_select_max, rvir_select_detect,\
impact_select_min, impact_select_max, impact_select_detect,\
coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element = element)

yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element)
plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color='tab:olive')
plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color='tab:olive')

element = 'Mg II'
gal_id_min, gal_id_max, gal_id_detect, \
rvir_select_min, rvir_select_max, rvir_select_detect,\
impact_select_min, impact_select_max, impact_select_detect,\
coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element = element)

yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
             fmt='o', color='tab:blue', label=r'$\rm N_{%s, obs}$'%element)
plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color='tab:blue')
plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color='tab:blue')

plt.xscale('log')
plt.yscale('log')
#plt.ylim(1e12, 10.**15.5)
plt.xlim(6e-2, 1.1)
plt.xlabel(r'$b/R_{vir}$')
plt.ylabel(r'Column density $[cm^{-2}]$')
plt.legend(loc='lower left', fancybox=True)
plt.grid()
plt.show()
'''

element = 'O VI'
gal_id_min, gal_id_max, gal_id_detect, \
rvir_select_min, rvir_select_max, rvir_select_detect,\
impact_select_min, impact_select_max, impact_select_detect,\
coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element = element)

plt.figure(figsize=(13,10))
yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
             fmt='o', color='tab:blue', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color='tab:blue', markersize=12)
plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color='tab:blue', markersize=12)

#model
R_vir = 211.94 #kpc

NOVI_PIE = np.load('./isent/NOVI_PIE.npy')
NOVI_CIE = np.load('./isent/NOVI_CIE.npy')

plt.plot(NOVI_PIE[:,0]/R_vir, np.nan_to_num(NOVI_PIE[:,1]), color='tab:blue', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NOVI_CIE[:,0]/R_vir, np.nan_to_num(NOVI_CIE[:,1]), color='tab:blue', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

element = 'N V'
gal_id_min, gal_id_max, gal_id_detect, \
rvir_select_min, rvir_select_max, rvir_select_detect,\
impact_select_min, impact_select_max, impact_select_detect,\
coldens_min, coldens_max, coldens_detect, e_coldens_detect = observedColDens().col_densGen(element = element)

yerr = np.log(10)*e_coldens_detect*10.**coldens_detect
plt.errorbar(impact_select_detect/rvir_select_detect, 10.**coldens_detect, yerr=yerr, 
             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.plot(impact_select_min/rvir_select_min, 10.**coldens_min, '^', color='tab:olive', markersize=12)
plt.plot(impact_select_max/rvir_select_max, 10.**coldens_max, 'v', color='tab:olive', markersize=12)

#model
NNV_PIE = np.load('./isent/NNV_PIE.npy')
NNV_CIE = np.load('./isent/NNV_CIE.npy')

plt.plot(NNV_PIE[:,0]/R_vir, np.nan_to_num(NNV_PIE[:,1]), color='tab:olive', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NNV_CIE[:,0]/R_vir, np.nan_to_num(NNV_CIE[:,1]), color='tab:olive', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

plt.xscale('log')
plt.yscale('log')
plt.ylim(10.**11.8, 10.**15.3)
plt.xlim(6e-2, 1.2)
plt.xlabel(r'$b/R_{vir}$', size=28)
plt.ylabel(r'Column density $[cm^{-2}]$' ,size=28)
leg = plt.legend(loc='lower left', ncol=3, fancybox=True, fontsize=25)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
leg.set_title("Isentropic profile with isochoric redistribution",prop={'size':20})
plt.savefig('./isent/ColumnDensity-warm.png', transparent=True)
plt.show()
plt.close()

plt.figure(figsize=(13,10))
'''
Reference: Miller Bregman 2013
Number	Name	l	b	EW	Error	$N_{{\rm O}\,{\scriptsize{VII}},{\rm thin}}$	Error	N_model, thin	$N_{{\rm O}\,{\scriptsize{VII}},{\rm saturated}}$	Error	N_model, saturated	
(deg)	(deg)	(mAring)	(mAring)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	(10^15 cm^-2)	
1	Mrk 421	179.83	65.03	11.8	0.8	4.12	2.53	4.62	5.36	3.61	4.54	
2	PKS 2155-304	17.73	-52.24	13.7	1.9	4.79	2.60	7.55	6.56	4.06	8.97	
'''

#model
element = 'O VII'
NOVII_PIE = np.load('./isent/NOVII_PIE.npy')
NOVII_CIE = np.load('./isent/NOVII_CIE.npy')

plt.plot(NOVII_PIE[:,0]/R_vir, np.nan_to_num(NOVII_PIE[:,1]), color='tab:blue', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NOVII_CIE[:,0]/R_vir, np.nan_to_num(NOVII_CIE[:,1]), color='tab:blue', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

#obs
NOVII_obs = 15.68 
NOVII_err = 0.27
yerr = np.log(10)*NOVII_err*10.**NOVII_obs
#plt.errorbar(np.array([6.5e-2,]), 2*10.**np.array([NOVII_obs,]), yerr=2*np.array([yerr,]), uplims=True,
#             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.axhspan(2*10.**NOVII_obs, 2*(10.**NOVII_obs-yerr), 
                 color='tab:blue', alpha=0.2, zorder=0)

element = 'O VIII'
NOVIII_PIE = np.load('./isent/NOVIII_PIE.npy')
NOVIII_CIE = np.load('./isent/NOVIII_CIE.npy')

plt.plot(NOVIII_PIE[:,0]/R_vir, np.nan_to_num(NOVIII_PIE[:,1]), color='tab:olive', label=r'$\rm N_{%s, PIE}$'%element, linewidth=5)
plt.plot(NOVIII_CIE[:,0]/R_vir, np.nan_to_num(NOVIII_CIE[:,1]), color='tab:olive', ls= ':', label=r'$\rm N_{%s, CIE}$'%element, linewidth=5)

#obs
NOVIII_obs = NOVII_obs-np.log10(4) 
NOVIII_err = NOVII_err-np.log10(4)
yerr = np.log(10)*NOVIII_err*10.**NOVIII_obs
#plt.errorbar(np.array([6.5e-2,]), 2*10.**np.array([NOVII_obs,]), yerr=2*np.array([yerr,]), uplims=True,
#             fmt='o', color='tab:olive', label=r'$\rm N_{%s, obs}$'%element, markersize=12)
plt.axhspan(2*10.**NOVIII_obs, 2*(10.**NOVIII_obs-yerr), 
                 color='tab:olive', alpha=0.2, zorder=0)

plt.xscale('log')
plt.yscale('log')
plt.ylim(10**14.0, 10.**16.3)
plt.xlim(6e-2, 1.2)
plt.xlabel(r'$b/R_{vir}$', size=28)
plt.ylabel(r'Column density $[cm^{-2}]$' ,size=28)
leg = plt.legend(loc='lower left', ncol=2, fancybox=True, fontsize=25)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=24)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=22)
plt.grid()
plt.tight_layout()
# set the linewidth of each legend object
# for legobj in leg.legendHandles:
leg.set_title("Isentropic profile with isochoric redistribution",prop={'size':20})
plt.savefig('./isent/ColumnDensity-hot.png', transparent=True)
plt.show()