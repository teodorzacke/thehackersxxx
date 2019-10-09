# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:50:46 2019

@author: caisa
"""

from scipy import *
from matplotlib.pyplot import *
from numpy import *

A = array([[1, 1, 2],
           [1, 2, 1],
           [2, 1, 1],
           [2, 2, 1]])

y = array([1,-1,1,-1])

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
        
x0 = array([1, 2, 3])   
#
#print(opt.fmin(leastsq2,x0))
#print(leastsq(A,y))

##point 3
#residual is basiaclly norm of Ax-y squared

def point3(M, a, lst=True):
    """
    Computes the least square method with an arbitrary a
    in the given vector y = (1,a,1,a). Third argument is callable,
    if you want both solution (minimizing vector) and
    the norm of the residual.
    """
    y2 = np.transpose(array([1,a,1,a]))
    lhs = np.matmul(np.transpose(M), M)
    rhs = np.matmul(np.transpose(M), y2)
    sol = np.linalg.solve(lhs, rhs)
    n = np.linalg.norm(np.dot(M,sol)-y2)
    if lst == True:
        return n
    else:
        return n and sol

A = array([[1, 1, 2],
           [1, 2, 1],
           [2, 1, 1],
           [2, 2, 1]])

xval = linspace(-100,100)
yval = [point3(A, a) for a in linspace(-100,100)]

plt.plot(xval, yval)
plt.title('The norm of the residual versus a in y = (1,a,1,a).')
plt.xlabel('value of a')
plt.ylabel('value of ||Ax-y||')
plt.grid()
plt.show()

"""
Does not have a value where the curve is zero.
We can see from the graph that the curve is increasing
in both negative and positive x direction
(i.e, it looks like y = |x|). This means, the interval
that needs to be checked is around a = 0.
"""

#for i in linspace(-2,2):
#    if point3(A, i) == 0:
#        print('A zero exists!', point3(A,i))
#        break
#    else:
#        print('A zero does not exist!')
#        break

        








