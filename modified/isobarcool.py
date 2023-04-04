# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:28:36 2022

@author: Alankar
"""
import sys
sys.path.append('..')
from modified.redistribution import Redistribution

class IsobarCoolRedistribution(Redistribution):

    def __init__(self, unmodifiedProfile, TmedVW=3.e5, sig=0.3, cutoff=2):
        super().__init__(unmodifiedProfile, TmedVW, sig, cutoff)
        self.isobaric = 1
