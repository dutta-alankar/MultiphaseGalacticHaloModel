# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:49:11 2021

@author: alankar
"""

import numpy as np
from typing import Union, Tuple


def toGalC(
    l: Union[float, int, list, np.ndarray],
    b: Union[float, int, list, np.ndarray],
    distance: Union[float, int, list, np.ndarray],
) -> tuple:
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
    L = np.deg2rad(np.array(l))
    B = np.deg2rad(np.array(b))
    Distance = np.array(distance)

    SuntoGC = 8.0  # kpc
    xgc = Distance * np.cos(B) * np.cos(L) - SuntoGC
    ygc = Distance * np.cos(B) * np.sin(L)
    zgc = Distance * np.sin(B)

    radius = np.sqrt(xgc**2 + ygc**2 + zgc**2)
    phi = np.rad2deg(np.arctan2(ygc, xgc))
    theta = np.rad2deg(np.arccos(zgc / radius))
    phi = np.select(
        [ygc > 0, ygc < 0], [phi, 360 + phi]
    )  # np.select([xgc*ygc<0,xgc*ygc>0], [phi+180, phi])

    return (radius, phi, theta)


def toSunC(
    phi: Union[float, list, np.ndarray],
    theta: Union[float, list, np.ndarray],
    distance: Union[float, list, np.ndarray],
) -> tuple:
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
    Phi = np.array(phi)
    Theta = np.array(theta)
    Distance = np.array(distance)

    Phi = np.deg2rad(Phi)
    Theta = np.deg2rad(Theta)
    SuntoGC = -8.0  # kpc
    xsc = Distance * np.sin(Theta) * np.cos(Phi) - SuntoGC
    ysc = Distance * np.sin(Theta) * np.sin(Phi)
    zsc = Distance * np.cos(Theta)

    radius = np.sqrt(xsc**2 + ysc**2 + zsc**2)
    phi = np.rad2deg(np.arctan2(ysc, xsc))
    theta = np.rad2deg(np.arccos(zsc / radius))
    l = np.array(phi)
    b = 90 - np.array(theta)

    return (radius, l, b)


def AngFromGC(
    l: Union[float, list, np.ndarray], b: Union[float, list, np.ndarray]
) -> Tuple[Union[float, list, np.ndarray]]:
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
    L = np.array(l)
    B = np.array(b)
    return np.rad2deg(np.arccos(np.cos(np.deg2rad(L)) * np.cos(np.deg2rad(B))))
