#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:51:48 2019

@author: teodor
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


### LINALG task 2

### 2.1
iterates = 100
A = np.array([[1,3,2],
              [-3,4,3],
              [2,3,1]])

z0 = np.array([8,3,12])
z01 = z0
listt1 = [z0]
for i in range(iterates):
    newterm = np.dot(A, z01)
    listt1.append(newterm)
    z01 = newterm




### 2.2
z02 = z0
listt2 = []
for i in listt1:
    listt2.append(i / np.linalg.norm(i))
cz = listt2[-1]

diffl = []
for i in range(len(listt2)-1):
    diffl.append(np.linalg.norm(listt2[i] - listt2[i-1]))
    
#X, Y, Z = list(zip(*listt2))
#fig = plt.figure()
#ax = p3.Axes3D(fig)
#ax.plot(X,Y,Z)
#plt.plot(range(iterates+1), X)
#plt.plot(range(iterates+1), Y)
#plt.plot(range(iterates+1), Z)

### 2.3
#listt3 = []
#for i in listt2:
#    listt3.append(np.dot(np.dot(np.transpose(i), A), i))

### 2.4
#z04 = z0
#count = 0
#while np.linalg.norm(z04 - cz) >= 10 ** -8:
#    newterm = np.dot(A, z04)
#    z04 = newterm
#    count += 1
#print(count)
for i in range(len(listt1)):
    if np.linalg.norm(listt1[i] - cz) < 10 ** -8:
        print(i)