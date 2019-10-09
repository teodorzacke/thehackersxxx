# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:55:27 2019

@author: Caisa
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def f(x,y,z):
    return x+2*y+z+math.exp(2*z)-1
#
def f_z(x,y):
    """
    Function that estimates a value for z such that f(x,y,z)=0. This is done by suggesting 
    different x,y values.
    Trying different values of the second argument in the fsolve function yields a sort of push in a direction on the x-axis
    which the function is shown (an interpretation by experimenting and observation). If a value -3<x0<3 is chosen, 
    a vertical plane will start to show. This is python's way of plotting functions where they are not defined/not continous
    (compare how it looks in geogebra)
    """
    def f2(z):
        return f(x,y,z)
    return fsolve(f2, 4)

x = np.linspace(-1,1,50)
y = np.linspace(-1,1,50)
X,Y = np.meshgrid(x,y)
z=[]
for (x,y) in zip(X,Y):
    z.append([f_z(i,j) for (i,j) in zip(x,y)])
Z = np.squeeze(np.asarray(z))                       #squeezing to get rid of the third dimension, plot_surface only takes 2D-arrays as args.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z)

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
















#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
