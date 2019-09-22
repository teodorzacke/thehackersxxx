# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:02:44 2019

@author: caisa
"""

from scipy import *
from matplotlib.pyplot import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin

x = arange(-1., 3., 0.1)
y = arange(-1., 5., 0.1)

def func(x, y):
    return 8*x*y-4*(x**2)*y-2*x*(y**2)+(x**2)*(y**2)

X, Y = meshgrid(x, y)
lvl = [i for i in range(0, 8)]
Z = func(X, Y)
figure()
con = contourf(X, Y, Z, lvl)
colorbar(con)
show()



