# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:10:12 2024

@author: alankar
"""

import numpy as np
from misc.coolLambda import cooling_approx
import matplotlib.pyplot as plt
import matplotlib
from decimal import Decimal

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

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
matplotlib.rcParams["ytick.major.width"] = 2.0
matplotlib.rcParams["xtick.major.width"] = 2.0
matplotlib.rcParams["ytick.minor.width"] = 1.0
matplotlib.rcParams["xtick.minor.width"] = 1.0
matplotlib.rcParams["ytick.major.size"] = 10.0
matplotlib.rcParams["xtick.major.size"] = 10.0
matplotlib.rcParams["ytick.minor.size"] = 6.0
matplotlib.rcParams["xtick.minor.size"] = 6.0
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
matplotlib.rcParams["figure.figsize"] = (15,10)
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

Temperature = np.logspace(1.2, 8.8, 1000)
metallicity = [0.3, 0.5, 1.0, 3.0]
colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

for indx, met_val in enumerate(metallicity):
    cloudy_pie = np.loadtxt("./cooltables/cooltable_PIE_Z=%.2f.dat"%met_val)
    # cloudy_cie = np.loadtxt("./cooltables/cooltable_CIE_Z=%.2f.dat"%met_val)
    
    approx_est = cooling_approx(Temperature, met_val)
    
    lines = plt.loglog(cloudy_pie[:,0], cloudy_pie[:,1], linestyle="-", 
                       linewidth=3, color=colors[indx],
                       label=r"$Z/Z_{\odot}=$%.1f"%met_val)
    # lines = plt.loglog(cloudy_cie[:,0], cloudy_cie[:,1], linestyle=":", color=lines[-1].get_color())
    lines = plt.loglog(Temperature, approx_est, color=lines[-1].get_color(), 
                       linewidth=3, linestyle=(0, (3, 5, 1, 5, 1, 5)))

lgd1 = plt.legend(
        loc = "upper right",
        prop={"size": 23}, 
        framealpha=0.9, shadow=False, fancybox=True,
        ncol=2,
        bbox_to_anchor=[0.95, 0.99],
    )
plt.gca().add_artist(lgd1)
line1, = plt.plot([], [], color="k", label=r"approximate prescription", 
                  linewidth=3, linestyle=(0, (3, 5, 1, 5, 1, 5)))
line2, = plt.plot([], [], color="k", label=r"$\tt{Cloudy}$ generated", 
                  linewidth=3, linestyle="-")

lgd2 = plt.legend(handles=[line1, line2,], loc = "lower right",
    prop={"size": 20}, 
    framealpha=0.9, shadow=False, fancybox=True,
    ncol=1,
    bbox_to_anchor=[0.99, 0.01],
)
plt.gca().add_artist(lgd2)

plt.xlabel(r"Gas Temperature [K]")
plt.ylabel(r"Cooling function $\Lambda=\dot{e}/n_H^2$ [$\rm erg\ cm^3\ s^{-1}$]")

plt.xlim(xmin=9.0e+03, xmax=9.0e+08)
plt.ylim(ymin=5.0e-24, ymax=9.0e-22)
plt.savefig("./cool-approx.png", bbox_inches='tight', transparent=False)
plt.show()

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
matplotlib.rcParams["ytick.major.width"] = 2.0
matplotlib.rcParams["xtick.major.width"] = 2.0
matplotlib.rcParams["ytick.minor.width"] = 1.0
matplotlib.rcParams["xtick.minor.width"] = 1.0
matplotlib.rcParams["ytick.major.size"] = 10.0
matplotlib.rcParams["xtick.major.size"] = 10.0
matplotlib.rcParams["ytick.minor.size"] = 6.0
matplotlib.rcParams["xtick.minor.size"] = 6.0
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
matplotlib.rcParams["figure.figsize"] = (15,10)
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True

met_val = 1.0
nH = [1.0e-02, 1.0e-03, 1.0e-04, 2.0e-05]
colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"]

for indx, nH_val in enumerate(nH):
    cloudy_pie = np.loadtxt( "./cooltables/cooltable_PIE_Z=%.2f-nH=%.1e.dat"%(met_val, nH_val) ) 
    # cloudy_cie = np.loadtxt( "./cooltables/cooltable_CIE_Z=%.2f-nH=%.1e.dat"%(met_val, nH_val) ) 
    lines = plt.loglog(cloudy_pie[:,0], cloudy_pie[:,1], 
                       linestyle="-", linewidth=3, color=colors[indx],
                       label=r"$n_H=%.1f \times 10^{%d}$ $\rm cm^{-3}$"%(fman(nH_val), fexp(nH_val)) )
    # lines = plt.loglog(cloudy_cie[:,0], cloudy_cie[:,1], color=lines[-1].get_color(), 
    #                    linestyle=(0, (3, 5, 1, 5, 1, 5)))

plt.legend(prop={"size": 23}, 
           framealpha=0.9, shadow=False, fancybox=True,
           ncol=2,
           loc="lower right",
)

plt.xlabel(r"Gas Temperature [K]")
plt.ylabel(r"Cooling function $\Lambda=\dot{e}/n_H^2$ [$\rm erg\ cm^3\ s^{-1}$]")

plt.xlim(xmin=9.0e+03, xmax=9.0e+08)
plt.ylim(ymin=5.0e-24, ymax=9.0e-22)
plt.savefig("./cool-nH_dep.png", bbox_inches='tight', transparent=False)
plt.show()