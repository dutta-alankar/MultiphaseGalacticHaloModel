#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:49:11 2021

@author: alankar
"""

import numpy as np

def toGalC(l, b, distance):
    """
    Convert galactocentric position wrt Sun to that wrt Galaxy center

    Parameters
    ----------
    l : float/list/numpy array
        Galactic Coordinate longitude in degree.
    b : float/list/numpy array
        Galactic Coordinate latitude in degree.
    distance : float/list/numpy array
        distance of the point of interest from Sun.

    Returns
    -------
    radius : float/list/numpy array
        distance of the point of interest from Galaxy center.
    phi : float/list/numpy array
        Azimuthal angle in degree wrt Galaxy center.
    theta : float/list/numpy array
        Polar angle in degree wrt Galaxy center.

    """
    if isinstance(l, list) : 
        L = np.array(l)
    elif isinstance(l, np.ndarray):
        L = np.copy(l)
    else:
        L = l
        
    if isinstance(b, list) : 
        B = np.array(b)
    elif isinstance(b, np.ndarray):
        B = np.copy(b)
    else:
        B = b
        
    if isinstance(distance, list) : 
        Distance = np.array(distance)
    elif isinstance(distance, np.ndarray):
        Distance = np.copy(distance)
    else:
        Distance = distance
            
    phi   = np.deg2rad(L)
    theta = np.pi/2-np.deg2rad(B)
    SuntoGC = 8.0 #kpc
    xgc = Distance*np.sin(theta)*np.cos(phi) - SuntoGC
    ygc = Distance*np.sin(theta)*np.sin(phi)
    zgc = Distance*np.cos(theta)
    
    radius = np.sqrt(xgc**2 + ygc**2 + zgc**2)
    phi    = np.rad2deg(np.arctan2(ygc, xgc))
    theta  = np.rad2deg(np.arccos(zgc/radius)) 
    phi    = np.select([ygc>0,ygc<0], [phi, 360+phi])   #np.select([xgc*ygc<0,xgc*ygc>0], [phi+180, phi])   
    
    return (radius, phi, theta)

def toSunC(phi, theta, distance):
    """
    Convert galactocentric position wrt Galaxy center to Sun

    Parameters
    ----------
    phi : float/list/numpy array
        Galactic Coordinate longitude in degree.
    theta : float/list/numpy array
        Galactic Coordinate latitude in degree.
    distance : float/list/numpy array
        distance of the point of interest from Sun.

    Returns
    -------
    radius : float/list/numpy array
        distance of the point of interest from Galaxy center.
    l : float/list/numpy array
        l in degree wrt Solar center.
    b : float/list/numpy array
        b in degree wrt Solar center.

    """
    if isinstance(phi, list) : 
        Phi = np.array(phi)
    elif isinstance(phi, np.ndarray):
        Phi = np.copy(phi)
    else:
        Phi = phi
        
    if isinstance(theta, list) : 
        Theta = np.array(theta)
    elif isinstance(theta, np.ndarray):
        Theta = np.copy(theta)
    else:
        Theta = theta
        
    if isinstance(distance, list) : 
        Distance = np.array(distance)
    elif isinstance(distance, np.ndarray):
        Distance = np.copy(distance)
    else:
        Distance = distance
        
    Phi   = np.deg2rad(Phi)
    Theta = np.deg2rad(Theta)
    SuntoGC = -8.0 #kpc
    xsc = Distance*np.sin(Theta)*np.cos(Phi) - SuntoGC
    ysc = Distance*np.sin(Theta)*np.sin(Phi)
    zsc = Distance*np.cos(Theta)
    
    radius = np.sqrt(xsc**2 + ysc**2 + zsc**2)
    phi    = np.rad2deg(np.arctan2(ysc, xsc))
    theta  = np.rad2deg(np.arccos(zsc/radius)) 
    l = phi
    b = 90-theta
    
    return (radius, l, b)

def AngFromGC(l, b):
    """
    Calculates angular distance from galactic center for a given (l,b)

    Parameters
    ----------
    l : float/list/numpy array
        Galactic Coordinate longitude in degree.
    b : float/list/numpy array
        Galactic Coordinate latitude in degree.

    Returns
    -------
    float/list/numpy array
        angular distance from galactic center in degree.

    """
    if isinstance(l, list) : 
        L = np.array(l)
    elif isinstance(l, np.ndarray):
        L = np.copy(l)
    else:
        L = l
        
    if isinstance(b, list) : 
        B = np.array(b)
    elif isinstance(b, np.ndarray):
        B = np.copy(b)
    else:
        B = b
    return np.rad2deg( np.arccos( np.cos(np.deg2rad(L)) * np.cos(np.deg2rad(B)) ) )