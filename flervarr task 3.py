# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:15:13 2019

@author: caisa
"""

import scipy as sc
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


#POINT 1

def f(x,y,z):
    return x+2*y+z+math.exp(2*z)-1
#
def f_z(x,y):
    """
    Function that estimates a value for z such that f(x,y,z)=0. This is done by suggesting 
    different x,y values. So basically, our z(x,y).
    Trying different values of the second argument in the fsolve function yields a sort of push in a direction on the x-axis
    which the function is shown (an interpretation by experimenting and observation).
    """
    def f2(z):
        return f(x,y,z)
    return fsolve(f2, 0)

#x = np.linspace(-1,1,50)
#y = np.linspace(-1,1,50)
#X,Y = np.meshgrid(x,y)
#z=[]
#for (x,y) in zip(X,Y):
#    z.append([f_z(i,j) for (i,j) in zip(x,y)])
#Z = np.squeeze(np.asarray(z))                       #squeezing to get rid of the third dimension, plot_surface only takes 2D-arrays as args.
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X,Y,Z)
#
#ax.set_xlabel('x axis')
#ax.set_ylabel('y axis')
#ax.set_zlabel('z axis')


#POINT 2

def num_der1(f, x, y, h=1e-8):
    """
    differentiate function with the definition of the derivative.
    f = callable function
    x, y = coordinate you want to evaluate at
    h = a very small number
    returns a tuple of the derivatives wrt x (1) and y (2)
    """
    return ((f(x+h,y)-f(x,y))/h),((f(x,y+h)-f(x,y))/h)

def num_der2(f, x, y, h=1e-4):
    """
    second derivative of function. Same input/output at num_der1.
    Why do we need 1e-8 on 11 and 22 derivatives on first term???
    """
    a = num_der1(f, x+h, y)[0]
    b = num_der1(f, x, y, h)[0]
    wrtx11 = (a-b)/h
    
    c = num_der1(f, x, y+h)[1]
    d = num_der1(f, x, y, h)[1]
    wrty22 = (c-d)/h
    
    e = num_der1(f, x, y+h, h)[0]
    f = num_der1(f, x, y, h)[0]
    mixed12 = (e-f)/h
#    wrtx = (f(x+2*h, y)-2*f(x+h, y)+f(x,y))/h**2
#    wrty = (f(x,y+2*h)-2*f(x,y+h)+f(x,y))/h**2
#    mixed = (f(x+h,y+h)-f(x,y+h)-f(x+h,y)-f(x,y))/h**2
    return wrtx11, mixed12, wrty22

#print(num_der2(f_z, 0, 0))


#POINT 3

def p2(f, x, y):
    """
    The Taylor polynomial around x=y=0 (Macalurin formula)
    with coefficients from num_der.
    """
    f1 = num_der1(f, x, y)[0]
    f2 = num_der1(f, x, y)[1]
    f11 = num_der2(f, x, y)[0]
    f12 = num_der2(f, x, y)[1]
    f22 = num_der2(f, x, y)[2]
    return float(f1*x+f2*y+f11*x**2+f12*x*y+f22*y**2)



#x = np.linspace(-1,1,15)
#y = np.linspace(-1,1,15)
#X,Y = np.meshgrid(x,y)
#z=[]
#for (x,y) in zip(X,Y):
#    z.append([p2(f_z,i,j) for (i,j) in zip(x,y)])
#Z = np.squeeze(np.asarray(z))                    #squeezing to get rid of the third dimension, plot_surface only takes 2D-arrays as args.
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X,Y,Z)
#ax.set_xlabel('x axis')
#ax.set_ylabel('y axis')
#ax.set_zlabel('z axis')



#POINT 4

def error(f, x, y):
    """
    Function of the error, defined in the assignment as 
    |z(x,y)-p2(x,y)|/|z(x,y)|
    """
    return float(abs(f(x,y)-p2(f,x,y))/abs(f(x,y)))

x = np.linspace(-1,1,15)
y = np.linspace(-1,1,15)
X,Y = np.meshgrid(x,y)
z=[]
for (x,y) in zip(X,Y):
    z.append([error(f_z,i,j) for (i,j) in zip(x,y)])
Z = np.squeeze(np.asarray(z))                    #squeezing to get rid of the third dimension, plot_surface only takes 2D-arrays as args.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')






