#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:29:25 2022

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import matplotlib.gridspec as gridspec

dispersionDisk = np.load('dispersionDisk.npy')

# Isothermal PIE
dispersion_isoth_ic_PIE = np.load('./isoth/dispersion_PIE_lb-ic.npy')

dispersionTot = dispersion_isoth_ic_PIE + dispersionDisk

b = np.linspace(-90, 90, 180)
l = np.linspace(0., 360, 360)

l, b = np.meshgrid(l, b)
    
# Make plot
levels = 100
l_plot = np.deg2rad(np.arange(0,360,45))
b_plot = np.deg2rad(np.arange(-90, -25, 30))

fig = plt.figure()#figsize=(20, 20))
gs = gridspec.GridSpec(1, 1)
# Position plot in figure using gridspec.
ax = plt.subplot(gs[0], polar=True)
ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

# Set x,y ticks
plt.xticks(l_plot)#, fontsize=8)
plt.yticks(b_plot)#, fontsize=8)
ax.set_rlabel_position(12)
#ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
#    '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
ax.set_yticklabels(['', '$-60^{\circ}$', '$-30^{\circ}$'])#, fontsize=10)
ax.set_theta_zero_location('S')
cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), dispersionTot, levels=levels, cmap='YlOrRd_r')
cbar = fig.colorbar(cs, pad = 0.08)
cbar.set_label(r'DM ($\rm cm^{-3} pc$)', rotation=270, labelpad=18, fontsize=18)
fig.tight_layout()
# plt.savefig('./DM-isth-ic_PIE.png') #, transparent=True, bbox_inches='tight')
plt.show()

l_mod = l-360 #np.select([l<=180,l>180],[l,l+360])
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(False)
cs = ax.pcolormesh(np.deg2rad(l_mod), np.deg2rad(b),  dispersionTot,  cmap='inferno') #, norm=colors.LogNorm())
cs = ax.pcolormesh(np.deg2rad(-l_mod), np.deg2rad(b), dispersionTot, cmap='inferno') #, norm=colors.LogNorm())
ax.grid(True)
cbar = fig.colorbar(cs, pad = 0.08, orientation='horizontal', 
                    shrink=0.5, aspect=60, location='top',
                    format='%.1f')
cbar.ax.tick_params(labelsize=12, length=6, width=2)
cbar.set_label(r'$\rm DM_{Disk+CGM}$ $\rm [cm^{-3} pc]$', rotation=0, labelpad=8, fontsize=18)
fig.tight_layout()
plt.grid(color='tab:grey', linestyle='--', linewidth=1.0)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=12)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=10)
ax.tick_params(axis='x', colors='white')

import matplotlib.transforms
plt.setp( ax.xaxis.get_majorticklabels()) 
# Create offset transform by 5 points in x direction
# Matplotlib figures use 72 points per inch (ppi). 
# So to to shift something by x points, you may shift it by x/72 inch. 
dx = 0/72.; dy = -30/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

axes_labels = np.array(ax.get_xticks().tolist())
print(axes_labels)
axes_labels = list(np.round(np.rad2deg(np.select([axes_labels<0,axes_labels>0],[axes_labels+2*np.pi,axes_labels]))))
#axes_labels = [r'$%d^{{\fontsize{50pt}{3em}\selectfont{}\circ}}$'%label for label in axes_labels]
axes_labels = [r'$%d^{\circ}$'%label for label in axes_labels]
print(axes_labels)
ax.set_xticklabels(axes_labels)
   
plt.savefig('./isoth/DM_moll-ic_PIE.png', transparent=True, bbox_inches='tight')
plt.show()


l_mod = l-360 #np.select([l<=180,l>180],[l,l+360])
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(False)
cs = ax.pcolormesh(np.deg2rad(l_mod),  np.deg2rad(b), dispersion_isoth_ic_PIE,  cmap='inferno')#, norm=colors.LogNorm())
cs = ax.pcolormesh(np.deg2rad(-l_mod), np.deg2rad(b), dispersion_isoth_ic_PIE,  cmap='inferno')#, norm=colors.LogNorm())
ax.grid(True)
cbar = fig.colorbar(cs, pad = 0.08, orientation='horizontal', 
                    shrink=0.5, aspect=60, location='top',
                    format='%.1f')
cbar.ax.tick_params(labelsize=12, length=6, width=2)
cbar.set_label(r'$\rm DM_{CGM}$ [$\rm cm^{-3} pc$]', rotation=0, labelpad=8, fontsize=18)
fig.tight_layout()
plt.grid(color='tab:grey', linestyle='--', linewidth=1.0)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=12)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=10)
ax.tick_params(axis='x', colors='white')

import matplotlib.transforms
plt.setp( ax.xaxis.get_majorticklabels()) 
# Create offset transform by 5 points in x direction
# Matplotlib figures use 72 points per inch (ppi). 
# So to to shift something by x points, you may shift it by x/72 inch. 
dx = 0/72.; dy = -30/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

axes_labels = np.array(ax.get_xticks().tolist())
# print(axes_labels)
axes_labels = list(np.round(np.rad2deg(np.select([axes_labels<0,axes_labels>0],[axes_labels+2*np.pi,axes_labels]))))
axes_labels = [r'$%d^{\circ}$'%label for label in axes_labels]
# print(axes_labels)
ax.set_xticklabels(axes_labels)
   
plt.savefig('./isoth/DM-only_CGM_moll-ic_PIE.png', transparent=True, bbox_inches='tight')
plt.show()

# Isentropic PIE
dispersion_isent_ic_PIE = np.load('./isent/dispersion_PIE_lb-ic.npy')

dispersionTot = dispersion_isent_ic_PIE + dispersionDisk

b = np.linspace(-90, 90, 180)
l = np.linspace(0., 360, 360)

l, b = np.meshgrid(l, b)
    
# Make plot
levels = 100
l_plot = np.deg2rad(np.arange(0,360,45))
b_plot = np.deg2rad(np.arange(-90, -25, 30))

fig = plt.figure()#figsize=(20, 20))
gs = gridspec.GridSpec(1, 1)
# Position plot in figure using gridspec.
ax = plt.subplot(gs[0], polar=True)
ax.set_ylim(np.deg2rad(-90), np.deg2rad(-25))

# Set x,y ticks
plt.xticks(l_plot)#, fontsize=8)
plt.yticks(b_plot)#, fontsize=8)
ax.set_rlabel_position(12)
#ax.set_xticklabels(['$22^h$', '$23^h$', '$0^h$', '$1^h$', '$2^h$', '$3^h$',
#    '$4^h$', '$5^h$', '$6^h$', '$7^h$', '$8^h$'], fontsize=10)
ax.set_yticklabels(['', '$-60^{\circ}$', '$-30^{\circ}$'])#, fontsize=10)
ax.set_theta_zero_location('S')
cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), dispersionTot, levels=levels, cmap='YlOrRd_r')
cbar = fig.colorbar(cs, pad = 0.08)
cbar.set_label(r'DM ($\rm cm^{-3} pc$)', rotation=270, labelpad=18, fontsize=18)
fig.tight_layout()
# plt.savefig('./DM-isth-ic_PIE.png') #, transparent=True, bbox_inches='tight')
plt.show()

l_mod = l-360 #np.select([l<=180,l>180],[l,l+360])
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(False)
cs = ax.pcolormesh(np.deg2rad(l_mod), np.deg2rad(b),  dispersionTot,  cmap='inferno') #, norm=colors.LogNorm())
cs = ax.pcolormesh(np.deg2rad(-l_mod), np.deg2rad(b), dispersionTot, cmap='inferno') #, norm=colors.LogNorm())
ax.grid(True)
cbar = fig.colorbar(cs, pad = 0.08, orientation='horizontal', 
                    shrink=0.5, aspect=60, location='top',
                    format='%.1f')
cbar.ax.tick_params(labelsize=12, length=6, width=2)
cbar.set_label(r'$\rm DM_{Disk+CGM}$ $\rm [cm^{-3} pc]$', rotation=0, labelpad=8, fontsize=18)
fig.tight_layout()
plt.grid(color='tab:grey', linestyle='--', linewidth=1.0)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=12)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=10)
ax.tick_params(axis='x', colors='white')

import matplotlib.transforms
plt.setp( ax.xaxis.get_majorticklabels()) 
# Create offset transform by 5 points in x direction
# Matplotlib figures use 72 points per inch (ppi). 
# So to to shift something by x points, you may shift it by x/72 inch. 
dx = 0/72.; dy = -30/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

axes_labels = np.array(ax.get_xticks().tolist())
print(axes_labels)
axes_labels = list(np.round(np.rad2deg(np.select([axes_labels<0,axes_labels>0],[axes_labels+2*np.pi,axes_labels]))))
#axes_labels = [r'$%d^{{\fontsize{50pt}{3em}\selectfont{}\circ}}$'%label for label in axes_labels]
axes_labels = [r'$%d^{\circ}$'%label for label in axes_labels]
print(axes_labels)
ax.set_xticklabels(axes_labels)
   
plt.savefig('./isent/DM_moll-ic_PIE.png', transparent=True, bbox_inches='tight')
plt.show()


l_mod = l-360 #np.select([l<=180,l>180],[l,l+360])
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(False)
cs = ax.pcolormesh(np.deg2rad(l_mod),  np.deg2rad(b), dispersion_isent_ic_PIE,  cmap='inferno')#, norm=colors.LogNorm())
cs = ax.pcolormesh(np.deg2rad(-l_mod), np.deg2rad(b), dispersion_isent_ic_PIE,  cmap='inferno')#, norm=colors.LogNorm())
ax.grid(True)
cbar = fig.colorbar(cs, pad = 0.08, orientation='horizontal', 
                    shrink=0.5, aspect=60, location='top',
                    format='%.1f')
cbar.ax.tick_params(labelsize=12, length=6, width=2)
cbar.set_label(r'$\rm DM_{CGM}$ [$\rm cm^{-3} pc$]', rotation=0, labelpad=8, fontsize=18)
fig.tight_layout()
plt.grid(color='tab:grey', linestyle='--', linewidth=1.0)
plt.tick_params(axis='both', which='major', length=12, width=3, labelsize=12)
plt.tick_params(axis='both', which='minor', length=8, width=2, labelsize=10)
ax.tick_params(axis='x', colors='white')

import matplotlib.transforms
plt.setp( ax.xaxis.get_majorticklabels()) 
# Create offset transform by 5 points in x direction
# Matplotlib figures use 72 points per inch (ppi). 
# So to to shift something by x points, you may shift it by x/72 inch. 
dx = 0/72.; dy = -30/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

axes_labels = np.array(ax.get_xticks().tolist())
# print(axes_labels)
axes_labels = list(np.round(np.rad2deg(np.select([axes_labels<0,axes_labels>0],[axes_labels+2*np.pi,axes_labels]))))
axes_labels = [r'$%d^{\circ}$'%label for label in axes_labels]
# print(axes_labels)
ax.set_xticklabels(axes_labels)
   
plt.savefig('./isent/DM-only_CGM_moll-ic_PIE.png', transparent=True, bbox_inches='tight')
plt.show()