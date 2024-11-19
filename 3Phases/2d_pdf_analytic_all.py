# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:21:48 2023

@author: Alankar
"""
import numpy as np
import h5py
import sys
import os
sys.path.append("..")
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from misc.coolLambda import cooling_approx
from scipy.integrate import simpson, trapezoid
from scipy.special import logsumexp
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse
from decimal import Decimal
import pickle

num_lum2D_calc = False

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent -1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

def lnprobT(y, params): # volume weighted volume PDF as a function of temperature
    fv    = params["fv"]
    sig   = params["sig"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    
    prob = np.array([(np.log(fv[i]/(2*sig[i,0]*sig[i,1]))) - \
                     0.5*np.log(np.pi*A[i]) + \
                     -(y-ymed[i])**2/(4.*A[i]*(sig[i,0]*sig[i,1])**2.) 
            for i in range(fv.shape[0])])
    return prob

def lnprobnH(x, params): # volume weighted volume PDF as a function of nH
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    A,B,C = params["factors"]
    print("Received: ", params)
    
    prob = np.array([np.log(fv[i]/(2*sig[i,0]*sig[i,1])) - \
                      0.5*np.log(np.pi*C[i]) + \
                      -(x-xmed[i])**2/(4.*C[i]*(sig[i,0]*sig[i,1])**2.) 
            for i in range(fv.shape[0])])
    return prob

def lnprob2D(x, y, params): # volume weighted volume PDF as a function of nH, temperature
    xx, yy = np.meshgrid(x, y)
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    
    prob = np.array([np.log(fv[i]/(2*np.pi*sig[i,0]*sig[i,1])) - (
                                                 A[i]*(xx-xmed[i])**2 +  
                                                 C[i]*(yy-ymed[i])**2 +
                                                 B[i]*(xx-xmed[i])*(yy-ymed[i]) )
        for i in range(fv.shape[0])])
    return {"hot" : prob[0,:,:],
            "warm": prob[1,:,:],
            "cold": prob[2,:,:],
            }

def mass_PDF(y, params): # mass weighted PDF as a function of temperature
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

# metallicity = 0.3

met_data = np.loadtxt("./Illustris-TNG50-1/met-trend.txt")
metallicity = interp1d(met_data[:,0], met_data[:,2], fill_value="extrapolate")

def lum_PDF(y, params): # luminosity weighted volume PDF as a function of temperature
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    T_medV_u = params["T_medV_u"]
    
    #yy = np.logspace(-2,2,1000)
    yy = np.copy(y)
    # xx = np.copy(x)
    # nH_val = np.exp(xx)*nH_medV_u
    # T_val = np.exp(yy)*T_medV_u
    met_val = np.power(10., metallicity(np.log10(np.exp(yy)*T_medV_u)))
    # np.savetxt("cloudy-cool-query.txt", np.vstack( (nH_val, T_val, met_val)).T)
    met_val[np.isnan(met_val)] = 1.0
    met_val = 0.3
    # print("metallicity: ", met_val)
    norm = np.array([fv[i]/(2.*sig[i,0]*sig[i,1]*np.sqrt(np.pi*A[i]))* \
                np.exp(2.*xmed[i] + 4.*C[i]*(sig[i,0]*sig[i,1])**2.)*\
                simpson(cooling_approx(np.exp(yy)*T_medV_u, met_val)*1.e22*\
                np.exp(-(yy-ymed[i]+2.*B[i]*(sig[i,0]*sig[i,1])**2.)**2./(4.*A[i]*(sig[i,0]*sig[i,1])**2.)),yy)
                for i in range(fv.shape[0])])
                 
    Norm = np.sum(norm)
   
    prob  = np.array([(fv[i]/(2.*sig[i,0]*sig[i,1]*np.sqrt(np.pi*A[i])) * \
                     np.exp(2.*xmed[i] + 4.*C[i]*(sig[i,0]*sig[i,1])**2.)*\
                     cooling_approx(np.exp(y)*T_medV_u, met_val)*1.e22 *\
                     np.exp(-(y-ymed[i]+2.*B[i]*(sig[i,0]*sig[i,1])**2.)**2./(4.*A[i]*(sig[i,0]*sig[i,1])**2.)))
            for i in range(fv.shape[0])])  
    return prob/Norm

def lnlum_PDF2D(x, y, params): # luminosity weighted volume PDF as a function of temperature
    fv    = params["fv"]
    sig   = params["sig"]
    xmed  = params["median_nH"]
    ymed  = params["median_T"]
    A,B,C = params["factors"]
    T_medV_u = params["T_medV_u"]
    
    dx = np.hstack( (x[1:]-x[:-1], x[-1]-x[-2]) )
    dy = np.hstack( (y[1:]-y[:-1], y[-1]-y[-2]) )
    
    xx, yy = np.meshgrid(x, y)
    dxx, dyy = np.meshgrid(dx, dy)
    nH_val = np.exp(xx)*nH_medV_u
    T_val = np.exp(yy)*T_medV_u
    met_val = np.power(10., metallicity(np.log10(np.exp(yy)*T_medV_u)))
    met_val[np.isnan(met_val)] = 1.0
    # print("Debug: ", x.shape, y.shape)
    cooling = np.loadtxt("./cooltable_PIE_request.dat")[:,-1].reshape(*xx.shape)
    # print("Debug: ", cooling.shape)
    lnPv2D = lnprob2D(x, y, params)
    lnPL2D = {}
    norm = 0
    for key in lnPv2D.keys():
        lnPL2D[key] = 2*xx + np.log(cooling) + lnPv2D[key] + 2*np.log(nH_medV_u)
        norm += np.sum(np.exp(lnPL2D[key])*dx*dy)
    lnPL2D = {f"{key}": (lnPL2D[key]-np.log(norm)) for key in lnPL2D.keys()}
    tmp = {}
    tmp["T"] = {f"{key}": np.log(np.sum(np.exp(lnPL2D[key])*dx, axis=1)) for key in lnPL2D.keys()}
    tmp["nH"] = {f"{key}": np.log(np.sum(np.exp(lnPL2D[key])*dy, axis=0)) for key in lnPL2D.keys()}
    for key in tmp.keys():
        lnPL2D[key] = tmp[key]

    return lnPL2D

f_Vh = 0.978
f_Vc = 0.0016
f_Vw = 1. - (f_Vh + f_Vc)
fV = np.array([f_Vh, f_Vw, f_Vc])

T_medV_u = 1.e5
T_h = 6.2e5
T_w = 8.0e4
T_c = 1.25e4
T_meds = np.array([T_h, T_w, T_c])

nH_medV_u = 1.e-5
nH_h = 1.4e-5
nH_w = 8.0e-5
nH_c = 1.1e-3
nH_meds = np.array([nH_h, nH_w, nH_c])

#  ln(nH/nHmedVu)    ln(T/TmedVu)
sig_h = [0.63, 0.4] # 0.8, 0.3]
sig_w = [0.56, 1.2] # [0.9, 0.6]
sig_c = [1.1, 0.15] # [1.1, 0.20]
sigs =  np.array([sig_h, sig_w, sig_c])         
               
alphas = np.array([19,55,176]) # [18,170,176]
cos_fac = np.cos(np.deg2rad(alphas))
sin_fac = np.sin(np.deg2rad(alphas))

A = np.array([ 0.5*( (cos_fac[i]/sigs[i,0])**2 + (sin_fac[i]/sigs[i,1])**2 )
               for i in range(alphas.shape[0])])
B = np.array([ sin_fac[i]*cos_fac[i]*( (1./sigs[i,0])**2 - (1./sigs[i,1])**2 )
               for i in range(alphas.shape[0])])
C = np.array([ 0.5*( (cos_fac[i]/sigs[i,1])**2 + (sin_fac[i]/sigs[i,0])**2 )
               for i in range(alphas.shape[0])])
print("Sanity: ", (np.product(sigs, axis=1)**2*(4*A*C-B**2)))

# change this to increase resolution
Temperature = np.logspace(3.99, 7.0, 128)
nH    = np.logspace(-6.0, -1.5, 129)
nH_val2D, T_val2D = np.meshgrid(nH, Temperature)
met_val = np.power(10., metallicity(np.log10(T_val2D.flatten())))
met_val[np.isnan(met_val)] = 1.0
nH_val2D, T_val2D = nH_val2D.flatten(), T_val2D.flatten()
np.savetxt("cloudy-cool-query.txt", np.vstack( (nH_val2D, T_val2D, met_val)).T)

xi    = np.log(nH_meds/nH_medV_u)
yi    = np.log(T_meds/T_medV_u)

data_dump = {"fV": fV,
             "T_meds": T_meds,
             "nH_meds": nH_meds,
             "sigs": sigs,
             "alphas": alphas,
             "ABC": [A, B, C],
             "Temperature": Temperature,
             "nH": nH,
             "xy_meds": [xi, yi],
             "T_medV_u": T_medV_u,
             "nH_medV_u": nH_medV_u,
             }
with open("params_data.pickle", "wb") as file_obj:
    pickle.dump(data_dump, file_obj)

median_temp = T_medV_u * np.exp(yi - B*(sigs[:,0]*sigs[:,1])**2.)
print("median temp are : ", median_temp)
print("volume filling fractions are : ", f_Vh, f_Vw, f_Vc)

lnpdf_vol_nHT = lnprob2D(np.log(nH/nH_medV_u), np.log(Temperature/T_medV_u),
                      {"fv": fV,
                       "sig": sigs,
                       "median_nH": xi,
                       "median_T": yi,
                       "factors": [A,B,C]
                      })

# log10_hot_pdf2D  = np.copy((lnpdf_vol_nHT["hot"]/np.log(10)))  + 2* np.log10(np.log(10)) # log10 PDF
# log10_warm_pdf2D = np.copy((lnpdf_vol_nHT["warm"]/np.log(10))) + 2* np.log10(np.log(10)) # log10 PDF
# log10_cold_pdf2D = np.copy((lnpdf_vol_nHT["cold"]/np.log(10))) + 2* np.log10(np.log(10)) # log10 PDF
# log10_total2D = logsumexp(np.array([log10_hot_pdf2D, log10_warm_pdf2D, log10_cold_pdf2D])*np.log(10),
#                           axis = 0)/np.log(10) # log10 PDF                       
# np.log10(10.**log10_hot_pdf2D +10.**log10_warm_pdf2D + 10.**log10_cold_pdf2D) 

ln_pdf_vol_T = lnprobT(np.log(Temperature/T_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_T": yi,
                   "factors": [A,B,C]
                   })
# log10_pdf_vol_T = ln_pdf_vol_T/np.log(10) + np.log10(np.log(10)) # log10 PDF in log10T

log10_pdf_vol_T = np.log10(np.exp(ln_pdf_vol_T)*np.log(10)) #+ np.log10(np.log(10)) # log10 PDF in log10T
tmp = np.power(10., log10_pdf_vol_T)
print("log10_pdf_vol_T: ", 
      np.trapz(tmp[0,:], np.log10(Temperature)),
      np.trapz(tmp[1,:], np.log10(Temperature)),
      np.trapz(tmp[2,:], np.log10(Temperature)))

ln_pdf_vol_nH = lnprobnH(np.log(nH/nH_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_nH": xi,
                   "factors": [A,B,C]
                   })
log10_pdf_vol_nH = np.log10(np.exp(ln_pdf_vol_nH)*np.log(10)) #+ np.log10(np.log(10)) # log10 PDF in log10nH
tmp = np.power(10., log10_pdf_vol_nH)
print("log10_pdf_vol_nH: ", 
      np.trapz(tmp[0,:], np.log10(nH)),
      np.trapz(tmp[1,:], np.log10(nH)),
      np.trapz(tmp[2,:], np.log10(nH)))
             
pdf_mass = mass_PDF(np.log(Temperature/T_medV_u), 
                  {"fv": fV,
                   "sig": sigs,
                   "median_nH": xi,
                   "median_T": yi,
                   "factors": [A,B,C]
                   }) * np.log(10)
                   
norm_mass = simpson(pdf_mass,np.log10(Temperature/T_medV_u))
print("mass normalization is :", np.sum(norm_mass))


pdf_lum = lum_PDF(np.log(Temperature/T_medV_u), 
                  # np.log(nH/nH_medV_u),
                  {"fv": fV,
                   "sig": sigs,
                   "median_nH": xi,
                   "median_T": yi,
                   "factors": [A,B,C],
                   "T_medV_u": T_medV_u,
                   }) * np.log(10)

if num_lum2D_calc:
    pdf_lum2D = lnlum_PDF2D(np.log(Temperature/T_medV_u), 
                            np.log(nH/nH_medV_u),
                            {"fv": fV,
                             "sig": sigs,
                             "median_nH": xi,
                             "median_T": yi,
                             "factors": [A,B,C],
                             "T_medV_u": T_medV_u,
                             }) 
    
    pdf_lum2D_log10 = {}
    for key in pdf_lum2D.keys():
        print("Debug: ", key)
        if key!="T" and key!="nH":
               pdf_lum2D_log10[key] = np.log10(np.exp(pdf_lum2D[key])*(np.log(10)**2))
        else:
           pdf_lum2D_log10[key] = {}
           for sub_key in pdf_lum2D[key].keys():
              pdf_lum2D_log10[key][sub_key] = np.log10(np.exp(pdf_lum2D[key][sub_key])*np.log(10))
              
    # norm_lum = simpson(pdf_lum,np.log10(Temperature/T_medV_u))
    # print("lum normalization is :", np.sum(norm_lum))                   
                   
    
## Plot Styling
sns.set(rc={"figure.figsize": (14, 6)})
sns.set_style("darkgrid", {"grid.linestyle": ":"})
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# plt.style.use('dark_background')
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = False
matplotlib.rcParams["ytick.right"] = False
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["axes.grid"] = False
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["legend.handletextpad"] = 0.4
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["lines.linewidth"] = 3.0
matplotlib.rcParams["ytick.major.width"] = 0.6
matplotlib.rcParams["xtick.major.width"] = 0.6
matplotlib.rcParams["ytick.minor.width"] = 0.45
matplotlib.rcParams["xtick.minor.width"] = 0.45
matplotlib.rcParams["ytick.major.size"] = 4.0
matplotlib.rcParams["xtick.major.size"] = 4.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["legend.handlelength"] = 2
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["figure.figsize"] = (14,6)
matplotlib.rcParams["axes.axisbelow"] = True

halo_id = 110
try:
    file = h5py.File(f"Illustris-TNG50-1/halo-prop_ID={halo_id}.hdf5", "r")
except FileNotFoundError:
    print("File unavailable!")
    sys.exit(1)

mass = np.log10(np.array(file["/NumberDensity"])) + np.log10(np.array(file["/Volume"]))
volume = np.log10(np.array(file["/Volume"]))

bins = 200
# grid plot limits set here for the 2D PDF
g = sns.JointGrid(xlim=(-5.8, -1.6), ylim=(3.9, 6.4), marginal_ticks=True)  # x = ,
Group_Pos = np.array(file["Group_Pos"])
Group_r200 = np.array(file["Group_r200"])

position = np.array(file["/Coordinates"])
distance = np.sqrt(np.sum(np.array([(position[:,i]-Group_Pos[i])**2 for i in range(position.shape[1])]).T, axis=1))

condition = np.logical_and(np.log10(file["/Temperature"]) >= 3.99, np.log10(file["/Temperature"]) <= 7.0)
condition = np.logical_and(condition, np.logical_and(np.log10(file["/nH"]) >= -6.0, np.log10(file["/nH"]) <= -1.5))
condition = np.logical_and(condition, np.array(file["/SFR"])<=1.0e-06)
condition = np.logical_and(condition, distance<=Group_r200)
'''
condition = np.logical_and(np.log10(file["/Temperature"]) >= 3.9, np.log10(file["/Temperature"]) <= 6.4)
condition = np.logical_and(condition, np.logical_and(np.log10(file["/nH"]) >= -6.0, np.log10(file["/nH"]) <= -1.5))
condition = np.logical_and(condition, np.array(file["/SFR"])<=1.0e-06)
'''

# counts, xedges, yedges, im = g.ax_joint.hist2d(
#     x=np.log10(file["/nH"])[condition],
#     y=np.log10(file["/Temperature"])[condition],
#     weights=volume[condition],
#     bins=(bins, bins),
#     density=True,
#     cmap="viridis",
#     norm=matplotlib.colors.LogNorm(),
#     zorder=-0.5,
# )

counts, xedges, yedges = np.histogram2d(
    x=np.log10(file["/nH"])[condition],
    y=np.log10(file["/Temperature"])[condition],
    weights=np.array(file["/Volume"])[condition],
    bins=(bins+1, bins),
    density=True,
)
# counts[counts<1.0e-02] = 0.
xcenters = 0.5*(xedges[1:]+xedges[:-1])
ycenters = 0.5*(yedges[1:]+yedges[:-1])
xxcenters, yycenters = np.meshgrid(xcenters, ycenters)

# g.ax_joint.contour(np.log10(nH), np.log10(Temperature), total, 
                   # colors='bisque', linestyles='-', alpha=0.8)
'''
g.ax_joint.contour(np.log10(nH), np.log10(Temperature), 10.**hot_pdf, 
                  colors='bisque', linestyles='-', alpha=0.8)
g.ax_joint.contour(np.log10(nH), np.log10(Temperature), 10.**warm_pdf, 
                  colors='bisque', linestyles='-', alpha=0.8)
g.ax_joint.contour(np.log10(nH), np.log10(Temperature), 10.**cold_pdf, 
                  colors='bisque', linestyles='-', alpha=0.8)
'''
sigs_plot = sigs/np.log(10) # (x,y) ---> (log10(nH), log10(T))

for num in range(1,3): # 1 sig and 2 sig
   ellipses = [Ellipse((np.log10(nH_meds[i]), np.log10(T_meds[i])), 
                       2*num*sigs_plot[i,0], 2*num*sigs_plot[i,1], 
                       angle=alphas[i], edgecolor = 'bisque', 
                       zorder = -0.2,
                       linewidth=2.0, fill=False) for i in range(3)]
   for ellipse in ellipses:
     g.ax_joint.add_artist(ellipse)

vol_pdf_tng, bin_edges = np.histogram(
    np.log10(file["/nH"])[condition],
    bins=bins+1,
    density=True,
    weights=np.array(file["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# top marginal
g.ax_marg_x.plot(np.log10(nH), log10_pdf_vol_nH[0,:],
                 color='indianred', linestyle=":", linewidth=1.0, label="hot")
g.ax_marg_x.plot(np.log10(nH), log10_pdf_vol_nH[1,:],
                 color='tab:green', linestyle=":", linewidth=1.0, label="warm")
g.ax_marg_x.plot(np.log10(nH), log10_pdf_vol_nH[2,:],
                 color='tab:blue', linestyle=":", linewidth=1.0, label="cold")

log10_pdf_vol_nH_tot = logsumexp(log10_pdf_vol_nH*np.log(10), axis=0)/np.log(10)
# np.log10(np.sum(10.**log10_pdf_vol_nH, axis=0))
g.ax_marg_x.plot(np.log10(nH), log10_pdf_vol_nH_tot,
                 color='darkorchid', linestyle="-", linewidth=1.0, label="total")
# data
g.ax_marg_x.plot(centers, 
                 np.log10(vol_pdf_tng), 
                 color="goldenrod",  alpha=1.0,
                 linestyle=":", zorder=-0.5)
g.ax_marg_x.plot(xcenters, 
                 np.log10(np.sum(counts.T, axis=0)*(yedges[1]-yedges[0])), 
                 color="goldenrod",  alpha=1.0,
                 label="data", zorder=-0.5)
np.savetxt("tmp-test.txt", np.vstack( (centers, vol_pdf_tng, xcenters, np.sum(counts.T, axis=0)*(yedges[1]-yedges[0])) ).T )
g.ax_marg_x.grid(linestyle=":", alpha=0.5)
g.ax_marg_x.set_ylim(ymin=-6.2, ymax=0.8)
g.ax_marg_x.text(-6.45, 1.0, r"$log_{10}$(PDF)", rotation=90, va='top')
for tick in g.ax_marg_x.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
for tick in g.ax_marg_x.xaxis.get_minor_ticks():
    tick.tick1line.set_visible(False)
ax_sec_x = g.ax_marg_x.secondary_xaxis("bottom", functions=(lambda x:10.**x, lambda x:np.log10(x)) )
ax_sec_x.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax_sec_x.set_xscale('log')
ax_sec_x.tick_params(axis="both", which="major", direction="in", length=7, width=1.2)
ax_sec_x.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
ax_sec_x.set_xticklabels([])

handles, labels = g.ax_marg_x.get_legend_handles_labels()
lgd = g.ax_marg_x.legend(handles, labels, loc='upper right', 
                         bbox_to_anchor=(1.25,1.4),
                         facecolor="white")
'''                         
lgd = g.ax_marg_x.legend(handles, labels, loc='upper right', 
                         bbox_to_anchor=(1.23,1.05),
                         facecolor="white")
'''
vol_pdf_tng, bin_edges = np.histogram(
    np.log10(file["/Temperature"])[condition],
    bins=bins,
    density=True,
    weights=np.array(file["/Volume"])[condition],
)
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# right marginal
g.ax_marg_y.plot(log10_pdf_vol_T[0,:], np.log10(Temperature),
                 color='indianred', linestyle=":", linewidth=1.0)
g.ax_marg_y.plot(log10_pdf_vol_T[1,:], np.log10(Temperature),
                 color='tab:green', linestyle=":", linewidth=1.0)
g.ax_marg_y.plot(log10_pdf_vol_T[2,:], np.log10(Temperature),
                 color='tab:blue', linestyle=":", linewidth=1.0)

log10_pdf_vol_T_tot = logsumexp(log10_pdf_vol_T*np.log(10), axis=0)/np.log(10)

logpdf_vol_nHT_tot = np.log10(np.log(10)*(np.exp(lnpdf_vol_nHT["hot"]) + 
                                          np.exp(lnpdf_vol_nHT["warm"]) + 
                                          np.exp(lnpdf_vol_nHT["cold"])))

im = g.ax_joint.pcolormesh((xedges[1:]+xedges[:-1])*0.5, (yedges[1:]+yedges[:-1])*0.5,
                counts.T, cmap="viridis",
                norm=matplotlib.colors.LogNorm(), zorder=-0.5)

cs = g.ax_joint.contour(np.log10(nH), np.log10(Temperature), logpdf_vol_nHT_tot,
                   np.linspace(-3, -0.1, 5),
                   colors="black", linewidths=1.0)
fmt = {}
strs = np.power(10., cs.levels)
for l, s in zip(cs.levels, strs):
    if fexp(s) < -1:
        fmt[l] = r"$%.0f\times 10^{%d}$"%(fman(s), fexp(s))
    else:
        fmt[l] = f"{s:.1f}"
clabels = g.ax_joint.clabel(cs, cs.levels, inline=True, colors="black", fontsize=10,
                  fmt=fmt)
for l in clabels:
    l.set_va("center")

'''
for txt in clabels:
    txt.set_backgroundcolor('w')
    txt.set_alpha(0.6)

cs = g.ax_joint.contour((xedges[1:]+xedges[:-1])*0.5, (yedges[1:]+yedges[:-1])*0.5, 
                        np.log10(counts.T),
                        np.linspace(-3, 0.1, 5),
                        colors="indianred", 
                        linestyles="-", linewidths=2.0)
fmt = {}
strs = np.power(10., cs.levels)
for l, s in zip(cs.levels, strs):
    fmt[l] = r"$%.0f\times 10^{%d}$"%(fman(s), fexp(s))
clabels = g.ax_joint.clabel(cs, cs.levels, inline=True, colors="indianred", 
                  fontsize=12.0,
                  fmt=fmt, zorder=500)

for txt in clabels:
    background = txt.set_backgroundcolor('cyan')
    print(dir(background))
    # txt.set_alpha(0.3)
'''

# load data
plot_ion = False
if plot_ion:
    data_ion = None
    element = "OVI"
    ion_loc = f"../submodules/AstroPlasma/misc/ionization_data-e{element}.pickle"
    if os.path.isfile(ion_loc):
        with open(ion_loc, "rb") as file_obj:
            data_ion = pickle.load(file_obj)
    
    frac_z02 = data_ion[f"f{element}_z02"]
    frac_z02[frac_z02<=-10.] = -10. 
    nH_ion = data_ion["nH"]
    temperature_ion = data_ion["temperature"]
    
    # Fraction contours
    cs = g.ax_joint.contour(np.log10(nH_ion), np.log10(temperature_ion), np.power(10.,frac_z02), 
                      [0.1,0.2, 0.3], 
                      colors="tab:red", linewidths=1.5)
    fmt = {}
    strs = cs.levels
    for l, s in zip(cs.levels, strs):
        fmt[l] = r"$%.1f$"%s
    clabels = g.ax_joint.clabel(cs, cs.levels, inline=True, colors="black", 
                      fontsize=12.0,
                      fmt=fmt, zorder=500)
       
# np.log10(np.sum(10.**log10_pdf_vol_T, axis=0))
g.ax_marg_y.plot( log10_pdf_vol_T_tot, np.log10(Temperature),
                 color='darkorchid', linestyle="-", linewidth=1.0, label="total")
# data
g.ax_marg_y.plot(np.log10(np.sum(counts.T, axis=1)*(xedges[1]-xedges[0])), 
                 ycenters, 
                 color="goldenrod", 
                 alpha=1.0, label="data", zorder=-0.5)
g.ax_marg_y.plot(np.log10(vol_pdf_tng), centers, color="goldenrod", 
                 alpha=1.0, linestyle=":", zorder=-0.5)
g.ax_marg_y.grid(linestyle=":", alpha=0.5)
g.ax_marg_y.set_xlim(xmin=-5.2, xmax=0.8)
g.ax_marg_y.text(-5.2, 3.81, r"$log_{10}$(PDF)", rotation=0, va='center')
for tick in g.ax_marg_y.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
for tick in g.ax_marg_y.yaxis.get_minor_ticks():
    tick.tick1line.set_visible(False)
ax_sec_y = g.ax_marg_y.secondary_yaxis("left", functions=(lambda x:10.**x, lambda x:np.log10(x)) )
ax_sec_y.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax_sec_y.set_yscale('log')
ax_sec_y.tick_params(axis="both", which="major", direction="in", length=7, width=1.2)
ax_sec_y.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
ax_sec_y.set_yticklabels([])

# colorbar
divider = make_axes_locatable(g.ax_marg_y)
cax = divider.append_axes("right", size="20%", pad=0.05)
cbar = plt.gcf().colorbar(im, cax=cax)
cbar.ax.yaxis.set_label_position("right")
cbar.ax.yaxis.tick_right()
cbar.ax.set_ylabel(r"Volume weighted distribution")
cbar.ax.tick_params('both', length=8, width=1.5, which='major', direction='out')
cbar.ax.tick_params('both', length=4, width=1.0, which='minor', direction='out')


g.ax_joint.set_ylim(ymin=3.99, ymax=6.4)
g.ax_joint.grid(linestyle=":", alpha=0.5, zorder=0.5)
g.ax_joint.set_axisbelow(True)
g.ax_joint.tick_params(axis="both", which="major", direction="in", length=7, width=1.2, zorder=11.5)
g.ax_joint.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8, zorder=11.5)
g.ax_joint.set_xlabel(r"Gas Hydrogen Density $n_H\ [log_{10}\ {\rm cm^{-3}}]$")
g.ax_joint.set_ylabel(r"Gas Temperature $[log_{10}\ {\rm K}]$")

ax_sec_y = g.ax_joint.secondary_yaxis("right", functions=(lambda x:10.**x, lambda x:np.log10(x)) )
ax_sec_y.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax_sec_y.set_yscale('log')
ax_sec_y.tick_params(axis="both", which="major", direction="in", length=7, width=1.2)
ax_sec_y.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
ax_sec_y.set_yticklabels([])

ax_sec_x = g.ax_joint.secondary_xaxis("top", functions=(lambda x:10.**x, lambda x:np.log10(x)) )
ax_sec_x.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax_sec_x.set_xscale('log')
ax_sec_x.tick_params(axis="both", which="major", direction="in", length=7, width=1.2)
ax_sec_x.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
ax_sec_x.set_xticklabels([])

# g.ax_joint.set_aspect("equal")
# g.ax_joint.sharey(g.ax_marg_y)

plt.savefig("./figures/distribution.png", bbox_inches="tight", transparent=False)
plt.show()
plt.close()

#####################################################################################
## Plot Styling
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
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
matplotlib.rcParams["figure.figsize"] = (13,10)
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

tng50 = np.loadtxt("./Illustris-TNG50-1/tng50-pdf-data.txt")
print("Plotting Illustris data!")
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 1],
    color="darkgoldenrod",
    linewidth=2.5,
    linestyle=":",
    alpha = 0.8,
)
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 2],
    color="yellowgreen",
    linewidth=2.5,
    linestyle=":",
    alpha = 0.8,
)
plt.plot(
    10.0 ** tng50[:, 0],
    tng50[:, 3],
    color="slateblue",
    linewidth=2.5,
    linestyle=":",
    alpha = 0.8,
)

mass_pdf_all = np.sum(pdf_mass, axis = 0)
lum_pdf_all  = np.sum(pdf_lum, axis = 0)
vol_pdf_all  = 10.**log10_pdf_vol_T_tot
pdf_vol_T = 10.**log10_pdf_vol_T
if num_lum2D_calc:
    lum_numerical = np.sum([np.power(10., pdf_lum2D_log10["T"][key]) for key in pdf_lum2D_log10["T"].keys()], axis=0)

# all phases
plt.plot(Temperature, vol_pdf_all, color="darkgoldenrod", label="volume PDF", linewidth=4)
plt.plot(Temperature, mass_pdf_all, color="yellowgreen", label="mass PDF", linewidth=4)
plt.plot(Temperature, lum_pdf_all, color="slateblue", label="luminosity PDF", linewidth=4)
if num_lum2D_calc: 
    plt.plot(Temperature, lum_numerical, color="slateblue", label="luminosity PDF (num)", linestyle="-.", linewidth=4)

# components
plt.plot(Temperature, pdf_vol_T[0,:], color="darkgoldenrod", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2)
plt.plot(Temperature, pdf_vol_T[1,:], color="darkgoldenrod", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2)
plt.plot(Temperature, pdf_vol_T[2,:], color="darkgoldenrod", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2)

# plt.plot(Temperature, pdf_mass[0,:], color="yellowgreen", linestyle="--", linewidth=1)
# plt.plot(Temperature, pdf_mass[1,:], color="yellowgreen", linestyle="--", linewidth=1)
# plt.plot(Temperature, pdf_mass[2,:], color="yellowgreen", linestyle="--", linewidth=1)

# plt.plot(Temperature, pdf_lum[0,:], color="slateblue", linestyle="--", linewidth=1)
# plt.plot(Temperature, pdf_lum[1,:], color="slateblue", linestyle="--", linewidth=1)
# plt.plot(Temperature, pdf_lum[2,:], color="slateblue", linestyle="--", linewidth=1)

print("luminosity fraction: hwc")
print(np.trapz(pdf_lum[0,:], np.log(Temperature/T_medV_u))/np.log(10),
      np.trapz(pdf_lum[1,:], np.log(Temperature/T_medV_u))/np.log(10),
      np.trapz(pdf_lum[2,:], np.log(Temperature/T_medV_u))/np.log(10) )

print("Mass fraction: hwc")
print(np.trapz(pdf_mass[0,:], np.log(Temperature/T_medV_u))/np.log(10),
      np.trapz(pdf_mass[1,:], np.log(Temperature/T_medV_u))/np.log(10),
      np.trapz(pdf_mass[2,:], np.log(Temperature/T_medV_u))/np.log(10) )

plt.xscale("log")
plt.yscale("log")
plt.ylim(10.0**-3.1, 10**1.1)
plt.xlim(10.0**3.90, 10.0**6.4)
plt.xlabel(r"Gas Temperature [$\rm K$]", size=28, color="black", labelpad=10)
plt.ylabel(r"$T \mathscr{P}(T)$", size=28, color="black", labelpad=10)
lgd1 = plt.legend(
        loc = "upper center",
        prop={"size": 20}, 
        framealpha=0.1, shadow=False, fancybox=True,
        ncol=3,
        bbox_to_anchor=[0.49, 0.97],
    )
plt.gca().add_artist(lgd1)

plt.tick_params(axis="both", which="major", length=15, width=2, labelsize=24)
plt.tick_params(axis="both", which="minor", length=8, width=1, labelsize=22)

line1, = plt.plot([], [], color="k", label="model prediction", linewidth=4)
line2, = plt.plot([], [], color="k", label="individual phases", linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2)
line3, = plt.plot([], [], color="k", linestyle=":", label="TNG50-1", linewidth=2.5)

lgd2 = plt.legend(handles=[line1, line2, line3], loc = "lower right",
    prop={"size": 20}, 
    framealpha=0.9, shadow=False, fancybox=True,
    ncol=1,
    bbox_to_anchor=[0.96, 0.02],
)
plt.gca().add_artist(lgd2)
plt.xlim(xmin=9.8e+03, xmax=2.5e+06)

plt.tight_layout()
#plt.grid()
# leg.set_title("Three phase PDF compared with a typical Illustris TNG50 Halo PDF", prop={'size':20})
plt.savefig("./figures/3-phases-pdf.png", transparent=False)
plt.show()
plt.close()
# revert back
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
