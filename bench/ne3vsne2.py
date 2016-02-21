# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:53:20 2016

@author: Robert A. McLeod
"""

import numpy as np
import numexpr3 as ne3

test = np.ones( [3,3 ] )

out = ne3.evaluate( "test+test" )
print( out )