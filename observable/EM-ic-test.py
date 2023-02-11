#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:03:04 2022

@author: alankar
"""

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import matplotlib.gridspec as gridspec
import observable.maps as maps
from misc.constants import *
from unmodified.isoth import IsothermalUnmodified
from unmodified.isent import IsentropicUnmodified
from modified.isochorcool import IsochorCoolRedistribution
from observable.EmissionMeasure import EmissionMeasure as EM

do_isothermal, do_isentropic = False, True
showProgress = False

# ____________________________________________________________
# _________________ Isothermal profile _______________________

if(do_isothermal):
    TmedVH=1.5e6
    TmedVW=3.e5
    sig = 0.3
    cutoff = 8.0
    THotM = TmedVH*np.exp(-sig**2/2)
    
    b = np.linspace(-89.8, -0.8, 40)
    l = np.linspace(0.2, 359.8, 21)
    
    l, b = np.meshgrid(l, b) 

    
    # PIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='PIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionM_PIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('emissionM_PIE_lb-isoth-ic.npy', emissionM_PIE)
    
    print('Saving plot ...')
    # Make plot.
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -0.8, 30))
    
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
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), emissionM_PIE*1e3, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3} cm^{-6} pc$)', rotation=270, labelpad=18)
    fig.tight_layout()
    plt.savefig('./EM-isth-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    # plt.show()
    
    l = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b),  emissionM_PIE*1e3, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(-b), emissionM_PIE*1e3, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3} cm^{-6} pc$)', rotation=270, labelpad=18)
    fig.tight_layout()
    plt.savefig('./EM_moll-isth-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    # plt.show()
    
    '''
    # CIE
    unmodified = IsothermalUnmodified(THot=THotM,
                          P0Tot=4580, alpha=1.9, sigmaTurb=60, 
                          M200=1e12, MBH=2.6e6, Mblg=6e10, rd=3.0, r0=8.5, C=12, 
                          redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionM_CIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('emissionM_CIE_lb-isoth-ic.npy', emissionM_CIE)
    
    
    print('Saving plot ...')
    # Make plot
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -2, 30))
    
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
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), emissionM_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3}\ cm^{-6} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./EM-isth-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()
    
    l_mod = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(b), emissionM_CIE, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(-b), emissionM_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3}\ cm^{-6} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./EM_moll-isth-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()
    '''
# ____________________________________________________________
# _________________ Isentropic profile _______________________

if(do_isentropic):
    nHrCGM = 1.1e-5
    TthrCGM = 1.5e6
    sigmaTurb = 60
    ZrCGM = 0.3
    TmedVW = 3.e5
    sig = 0.3
    cutoff = 8.0
    
    b = np.linspace(-89.8, -0.8, 40)
    l = np.linspace(0.2, 359.8, 21)
    
    l, b = np.meshgrid(l, b) 

    
    # PIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='PIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionM_PIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('emissionM_PIE_lb-isent-ic.npy', emissionM_PIE)
    
    
    print('Saving plot ...')
    # Make plot.
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -28, 30))
    
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
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), emissionM_PIE*1e3, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3} cm^{-6} pc$)', rotation=270, labelpad=18)
    fig.tight_layout()
    plt.savefig('./EM-isent-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    # plt.show()
    
    l = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b),  emissionM_PIE*1e3, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(-b), emissionM_PIE*1e3, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3} cm^{-6} pc$)', rotation=270, labelpad=18)
    fig.tight_layout()
    plt.savefig('./EM_moll-isent-ic_PIE.png') #, transparent=True, bbox_inches='tight')
    # plt.show()
    
    '''
    # CIE
    unmodified = IsentropicUnmodified(nHrCGM=nHrCGM, TthrCGM=TthrCGM, sigmaTurb=sigmaTurb, ZrCGM=ZrCGM,
                                      redshift=0.001, ionization='CIE')
    mod_isochor = IsochorCoolRedistribution(unmodified, TmedVW, sig, cutoff)
    
    emissionM_CIE = EM(mod_isochor).generate(l, b, showProgress=showProgress)
    np.save('emissionM_CIE_lb-isent-ic.npy', emissionM_CIE)
    
    
    print('Saving plot ...')
    # Make plot
    levels = 100
    l_plot = np.deg2rad(np.arange(0,360,45))
    b_plot = np.deg2rad(np.arange(-90, -2, 30))
    
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
    cs = ax.contourf(np.deg2rad(l), np.deg2rad(b), emissionM_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3}\ cm^{-6} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./EM-isent-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()
    
    l_mod = np.select([l<=180,l>180],[l,l-360])
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(b), emissionM_CIE, levels=levels, cmap='YlOrRd_r')
    cs = ax.contourf(np.deg2rad(l_mod), np.deg2rad(-b), emissionM_CIE, levels=levels, cmap='YlOrRd_r')
    cbar = fig.colorbar(cs, pad = 0.08)
    cbar.set_label(r'EM ($\rm \times 10^{-3}\ cm^{-6} pc$) [CIE]', rotation=270, labelpad=18, fontsize=18)
    fig.tight_layout()
    plt.savefig('./EM_moll-isent-ic_CIE.png') #, transparent=True, bbox_inches='tight')
    #plt.show()
    '''
