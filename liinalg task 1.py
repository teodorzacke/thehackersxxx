# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:50:46 2019

@author: caisa
"""

from scipy import *
from matplotlib.pyplot import *
from numpy import *

A = [(1,1,2),
     (1,2,1),
     (2,1,1),
     (2,2,1)]

y = [(1),
     (-1),
     (1),
     (-1)]

#method of least squares: At*A=Aty

lhs = np.matmul(np.transpose(A), A)
rhs = np.matmul(np.transpose(A), y)

sol = np.linalg.solve(lhs, rhs)

print(sol)
