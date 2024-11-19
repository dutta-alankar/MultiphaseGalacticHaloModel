#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:39:05 2023

@author: alankar
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import simpson, trapezoid
import sys
sys.path.append("..")
from misc.coolLambda import cooling_approx

def probT(y, params):
    fv    = params["fv"]
    sig   = params["sig"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    
    prob = np.array([(fv[i]/(2*sig[i,0]*sig[i,1])) / \
                     np.sqrt(np.pi*A[i]) * \
                     np.exp( -(y-ymed[i])**2/(4.*A[i]*(sig[i,0]*sig[i,1])**2.) )
            for i in range(fv.shape[0])])  
    return prob

def probnH(x, params):
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    A,B,C = params["factors"]
    
    prob = np.array([(fv[i]/(2*np.pi*sig[i,0]*sig[i,1])) * \
                      np.sqrt(np.pi/C[i]) * \
                      np.exp( -(x-xmed[i])**2/(4.*C[i]*(sig[i,0]*sig[i,1])**2.) )
            for i in range(fv.shape[0])])
    return prob

def mass_PDF(y, params):
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    
    norm  = np.array([fv[i] * np.exp(xmed[i] + C[i]*(sig[i,0]*sig[i,1])**2.)  
                      for i in range(fv.shape[0])])  
    Norm = np.sum(norm)
 
    prob  = np.array([fv[i] * np.exp(xmed[i] + C[i]*(sig[i,0]*sig[i,1])**2.)/ \
                     (2.*sig[i,0]*sig[i,1]*np.sqrt(np.pi*A[i])) *\
                     np.exp( -(y-ymed[i]+B[i]*(sig[i,0]*sig[i,1])**2.)**2./(4.*A[i]*(sig[i,0]*sig[i,1])**2.))
            for i in range(fv.shape[0])])  
            
    return prob/Norm
    
Z0 = 1.0
ZrCGM = 0.3
p = Z0 / ZrCGM
metallicity = (
    1.5
    * (p - (p**2 - 1) * np.arcsin(1.0 / np.sqrt(p**2 - 1)))
    * Z0
    * np.sqrt(p**2 - 1)
)

def lum_PDF(y, params):
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    T_medV_u = params["T_medV_u"]
    
    #yy = np.logspace(-2,2,1000)
    yy = np.copy(y)
    norm = np.array([fv[i]/(2.*sig[i,0]*sig[i,1]*np.sqrt(np.pi*A[i]))* \
                np.exp(2.*xmed[i] + 4.*C[i]*(sig[i,0]*sig[i,1])**2.)*\
                simpson(cooling_approx(np.exp(yy)*T_medV_u, metallicity)*1.e22*\
                np.exp(-(yy-ymed[i]+2.*B[i]*(sig[i,0]*sig[i,1])**2.)**2./(4.*A[i]*(sig[i,0]*sig[i,1])**2.)),yy)
                for i in range(fv.shape[0])])
                 
    Norm = np.sum(norm)
   
    prob  = np.array([(fv[i]/(2.*sig[i,0]*sig[i,1]*np.sqrt(np.pi*A[i])) * \
                     np.exp(2.*xmed[i] + 4.*C[i]*(sig[i,0]*sig[i,1])**2.)*\
                     cooling_approx(np.exp(y)*T_medV_u, metallicity)*1.e22 *\
                     np.exp(-(y-ymed[i]+2.*B[i]*(sig[i,0]*sig[i,1])**2.)**2./(4.*A[i]*(sig[i,0]*sig[i,1])**2.)))
            for i in range(fv.shape[0])])  
    return prob/Norm

#####################################################################################
## Plot Styling
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["legend.handletextpad"] = 0.4
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["lines.linewidth"] = 1.5
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.minor.width"] = 0.45
matplotlib.rcParams["xtick.minor.width"] = 0.45
matplotlib.rcParams["ytick.major.size"] = 4.0
matplotlib.rcParams["xtick.major.size"] = 4.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["xtick.major.pad"] = 10.0
matplotlib.rcParams["xtick.minor.pad"] = 10.0
matplotlib.rcParams["ytick.major.pad"] = 6.0
matplotlib.rcParams["ytick.minor.pad"] = 6.0
matplotlib.rcParams["xtick.labelsize"] = 24.0
matplotlib.rcParams["ytick.labelsize"] = 24.0
matplotlib.rcParams["axes.titlesize"] = 24.0
matplotlib.rcParams["axes.labelsize"] = 28.0
matplotlib.rcParams["axes.labelpad"] = 8.0
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

with open("params_data.pickle", "rb") as file_obj:
    data = pickle.load(file_obj)

fV = data["fV"]
T_meds = data["T_meds"]
nH_meds = data["nH_meds"]
sigs = data["sigs"]
alphas = data["alphas"]
A,B,C = data["ABC"]
Temperature = data["Temperature"]
nH = data["nH"]
xi, yi = data["xy_meds"]
T_medV_u = data["T_medV_u"]
nH_medV_u = data["nH_medV_u"]

pdf_vol_T = probT(np.log(Temperature/T_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_T": yi,
                   "factors": [A,B,C]
                   }) *np.log(10)
             
pdf_mass = mass_PDF(np.log(Temperature/T_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_nH": xi,
                   "median_T": yi,
                   "factors": [A,B,C]
                   }) *np.log(10)
                   
norm_mass = simpson(pdf_mass,np.log10(Temperature/T_medV_u))
print("mass normalization is :", np.sum(norm_mass))

pdf_lum = lum_PDF(np.log(Temperature/T_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_nH": xi,
                   "median_T": yi,
                   "factors": [A,B,C],
                   "T_medV_u": T_medV_u,
                   }) *np.log(10)
                   
norm_lum = simpson(pdf_lum,np.log10(Temperature/T_medV_u))
print("lum normalization is :", np.sum(norm_lum))                   
                   
pdf_vol_nH = probnH(np.log(nH/nH_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_nH": xi,
                   "factors": [A,B,C]
                   })*np.log(10)

pdf_vol_T_Tot = np.sum(pdf_vol_T, axis=0)

tng50 = np.loadtxt("./Illustris-TNG50-1/tng50-pdf-data.txt")
fig = plt.figure(figsize=(13, 10))
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 1],
    color="darkgoldenrod",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 2],
    color="yellowgreen",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 3],
    color="slateblue",
    linewidth=2,
    linestyle="--",
)

mass_pdf = np.sum(pdf_mass, axis = 0)
lum_pdf  = np.sum(pdf_lum, axis = 0)
vol_pdf  = np.copy(pdf_vol_T_Tot)

plt.plot(Temperature, vol_pdf, color="darkgoldenrod", label="volume PDF", linewidth=2)
plt.plot(Temperature, mass_pdf, color="yellowgreen", label="mass PDF", linewidth=2)
plt.plot(Temperature, lum_pdf, color="slateblue", label="luminosity PDF", linewidth=2)

plt.plot(Temperature,pdf_vol_T[0,:], color="darkgoldenrod", linestyle=":", linewidth=2)
plt.plot(Temperature,pdf_vol_T[1,:], color="darkgoldenrod", linestyle=":", linewidth=2)
plt.plot(Temperature,pdf_vol_T[2,:], color="darkgoldenrod", linestyle=":", linewidth=2)
plt.plot(Temperature, pdf_mass[0,:], color="yellowgreen", linestyle=":", linewidth=2)
plt.plot(Temperature, pdf_mass[1,:], color="yellowgreen", linestyle=":", linewidth=2)
plt.plot(Temperature, pdf_mass[2,:], color="yellowgreen", linestyle=":", linewidth=2)

plt.plot(Temperature, pdf_lum[0,:], color="slateblue", linestyle=":", linewidth=2)
plt.plot(Temperature, pdf_lum[1,:], color="slateblue", linestyle=":", linewidth=2)
plt.plot(Temperature, pdf_lum[2,:], color="slateblue", linestyle=":", linewidth=2)

plt.xscale("log")
plt.yscale("log")
plt.ylim(10.0**-3.1, 10**0.5)
plt.xlim(10.0**3.99, 10.0**6.4)
plt.tick_params(axis="both", which="major", length=15, width=1.5, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=1, labelsize=22)
# plt.yticks(fontsize=40)
# plt.xticks(fontsize=40)
plt.ylabel(r"$T \mathscr{P}(T)$", size=28, color="black", labelpad=35)
plt.xlabel(r"Temperature [$K$]", size=28, color="black", horizontalalignment='center')
leg = plt.legend(loc="best", ncol=1, fancybox=True, fontsize=20, framealpha=0.5)
plt.tick_params(axis="both", which="major", length=4, width=1, labelsize=10)
plt.tick_params(axis="both", which="minor", length=4, width=1, labelsize=10)
plt.tight_layout()
#plt.grid()
# leg.set_title("Three phase PDF compared with a typical Illustris TNG50 Halo PDF", prop={'size':20})
plt.savefig("./figures/3-phases-pdf.png")
plt.show()
plt.close()
