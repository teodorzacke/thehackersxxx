#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:03:49 2019

@author: teodor
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

### linalgtask2 ####

A = np.array([[1,3,2],
              [-3,4,3],
              [2,3,1]])
z0 = np.array([8,3,12])
eigva, eigve = sp.linalg.eig(A)[0], sp.linalg.eig(A)[1]
cve = np.dot(sp.linalg.inv(eigve), z0)

###

uk = lambda k: sp.dot(  sp.multiply(eigve, sp.power(eigva,k)), cve)
norm = lambda k: sp.multiply(uk(k), sp.power(sp.linalg.norm(uk(k)), -1))

###

xli = range(500)
zk = 19**(-0.5) * np.array([3,1,3]) ## gissar norm(inf) med min kompis wolfram
que = [np.dot(np.dot(np.transpose(norm(i)),A),norm(i)) for i in xli]

def pops(e):
    i = 0
    while sp.linalg.norm(norm(i)-zk) > e:
        i += 1
    return i    

X = [10**(-i) for i in np.linspace(1,15,15)]
#plt.plot([pops(i) for i in X], X)
plt.plot(range(500), [sp.linalg.norm(norm(i)-zk) for i in range(500)])