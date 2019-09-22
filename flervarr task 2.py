# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:04:11 2019

@author: caisa
"""

from scipy import integrate
from matplotlib.pyplot import *
from numpy import *

#Task 2
def func(x):
    return sqrt(4*x**2+9*x**4)

q = integrate.quad(func, -2, 1)
print(q)

