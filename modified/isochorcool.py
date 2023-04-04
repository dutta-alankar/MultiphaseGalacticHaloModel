# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:15:09 2022

@author: Alankar
"""
import sys
sys.path.append('..')
from modified.redistribution import Redistribution

class IsochorCoolRedistribution(Redistribution):

    def __init__(self, unmodifiedProfile, TmedVW=3.e5, sig=0.3, cutoff=2):
        super().__init__(unmodifiedProfile, TmedVW, sig, cutoff)
        self.isobaric = 0
