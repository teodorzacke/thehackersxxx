# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:50:46 2019

@author: caisa
"""

from scipy import *
from matplotlib.pyplot import *
from numpy import *

A = array([[1,1],
           [4,-1],
           [3,2]])

y = array([6,8,5])

z = array([1,2])

#method of least squares: At*A=Aty
#point 1
def leastsq(A, y):
    """
    Setup of functions equivalent to what was to bedone manually
    when computing the method of least squares.
    """
    lhs = np.matmul(np.transpose(A), A)
    rhs = np.matmul(np.transpose(A), y)
    sol = np.linalg.solve(lhs, rhs)
    return sol

#point 2
def leastsq2(x):
    """
    directly interpreting (Ax-y)**2 ==> (Ax-y)*(Ax-y)
    as a matrix multiplication (works exactly like the nympy dot product here)
    """
    return np.dot(np.dot(A,x)-y,np.dot(A,x)-y)
        
x0 = array([1, 2])   

print(opt.fmin(leastsq2,x0))
print(leastsq(A,y))

#point 3







