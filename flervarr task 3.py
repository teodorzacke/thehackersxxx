# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:55:27 2019

@author: Caisa
"""
import scipy as sc
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.optimize import fsolve


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


#POINT 2

def num_der2(f, x, y, k=1e-4):
    """
    Differentiate to second degree.
    
    INPUT
    f = callable function
    x, y = coordinate you want to evaluate at
    h = a very small number
    
    RETURN
    first der wrt x, first der wrt y, second der wrt xx, second der mixed, second der wrt yy
    (in that order)
    
    OBS! these are the numbers that are later inserted in the Taylor approx formula
    """
    def num_der1(f, x, y, h=1e-8):
        """
        differentiate function with the definition of the derivative.
        """
        return ((f(x+h,y)-f(x,y))/h),((f(x,y+h)-f(x,y))/h)
    h = 1e-8
    k = 1e-4

    xx = float((f(x+h+k,y)-f(x+k,y)-f(x+h,y)+f(x,y))/(h*k))
    yy = float((f(x,y+h+k)-f(x,y+k)-f(x,y+h)+f(x,y))/(h*k))
    xy = float((f(x+h,y+k)-f(x,y+k)-f(x+h,y)-f(x,y))/(h*k))
    x1 = float(num_der1(f, x, y, h)[0])
    y1 = float(num_der1(f, x, y, h)[1])
    
    return x1, y1, xx, xy, yy

#print(num_der2(f_z, 0, 0))

#POINT 3

def p2(f, x, y):
    """
    The Taylor polynomial around x=y=0 (Macalurin formula)
    with coefficients from num_der.
    
    f(x,y)~f(0,0)+<(x,y),nabla>f(0,0)+(<(x,y),nabla>)^2f(0,0)
            = 
    """
    f1 = num_der2(f_z, 0, 0)[0]
    f2 = num_der2(f_z, 0, 0)[1]
    f11 = num_der2(f_z, 0, 0)[2]
    f12 = num_der2(f_z, 0, 0)[3]
    f22 = num_der2(f_z, 0, 0)[4]
    return float(f1*x+f2*y+(1/2)*(f11*x**2+f12*x*y+f22*y**2))


x = np.linspace(-1,1,30)
y = np.linspace(-1,1,30)
X,Y = np.meshgrid(x,y)
z=[]
for (x,y) in zip(X,Y):
    z.append([p2(f_z,i,j) for (i,j) in zip(x,y)])
Z = np.squeeze(np.asarray(z))                   
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')



#POINT 4

def error(f, x, y):
    """
    Function of the error, defined in the assignment as 
    |z(x,y)-p2(x,y)|/|z(x,y)|
    """
    return float(abs(f(x,y)-p2(f,x,y))/abs(f(x,y)))

x = np.linspace(-1,1,30)
y = np.linspace(-1,1,30)
X,Y = np.meshgrid(x,y)
z=[]
for (x,y) in zip(X,Y):
    z.append([error(f_z,i,j) for (i,j) in zip(x,y)])
Z = np.squeeze(np.asarray(z))                    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')


