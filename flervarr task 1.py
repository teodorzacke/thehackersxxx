# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:04:34 2019

@author: caisa
"""

from scipy import *
from matplotlib.pyplot import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin

x = arange(-1., 3., 0.1)
y = arange(-1., 5., 0.1)

def func(x,y):
    return 8*x*y-4*(x**2)*y-2*x*(y**2)+(x**2)*(y**2)

def save_it(k):
    """
    callback function to retireve all values in opt.fmin
    """
    global xit
    global zit
    xit.append(k[0])
    zit.append(k[1])

xit = []  #a list with all the iteration values of fmin
zit = []

func2 = lambda x : -(8*x[0]*x[1]-4*x[0]**2*x[1]-2*x[0]*x[1]**2+x[0]**2*x[1]**2)
#makes function negative because it will find a max otherwise

guess = np.array([0,4])
xval = linspace(0,22)
opt.fmin(func2, guess, full_output=True, callback=save_it)

X, Y = meshgrid(x, y)
lvl = [i for i in range(0, 8)]
Z = func(X, Y)
guess = np.array([1,2])
con = contourf(X, Y, Z, lvl)
plot(xit, zit, color='black')
colorbar(con)
title('Contour Plot')
xlabel('x')
ylabel('y')
show()

print(opt.fmin(func2, guess))


