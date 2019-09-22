#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:46:04 2019

@author: teodor
"""

#FLERVAR UPPGIFTER
import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy import integrate
import matplotlib.pyplot as plt

func = lambda x, y : 8*x*y-4*x**2*y-2*x*y**2+x**2*y**2

xlist = np.linspace(-50,50)
ylist = np.linspace(-50,50)
X, Y = np.meshgrid(xlist, ylist)
Z = func(X, Y)

#cp = plt.contourf(X, Y, Z, 100)

# DEL 2 av 1
func2 = lambda x : -(8*x[0]*x[1]-4*x[0]**2*x[1]-2*x[0]*x[1]**2+x[0]**2*x[1]**2)


guess = [1,2]
print(opt.fmin(func2, guess))


# DEL 2
# Räkna ut längden av kurvan
f = lambda x: (4*x**2+x**6)**0.5
print(integrate.quad(f, -2, 1))

