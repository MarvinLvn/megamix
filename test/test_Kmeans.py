# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:35:18 2017

@author: Elina Thibeau-Sutre
"""
import numpy as np
from megamix import Kmeans
import time
from megamix.initializations import initialization_plus_plus,initialization_AF_KMC
from megamix.kmeans import dist_matrix

if __name__ == '__main__':
    
    
    
    n_points = 100000
    k = 100
    dim = 39
    n_iter = 100
    tol = 0
    
    means = np.empty((k,dim))
    log_prob = np.empty((n_points,k))
    data = np.random.randn(n_points,dim)
    test = np.random.randn(n_points,dim)
    
    KM = Kmeans(k,init='plus')
    t0 = time.time()
    KM.fit(data,tol=tol)
    t1 = time.time()
#    initialization_plus_plus(k,data)
    t2 = time.time()
#    initialization_AF_KMC(k,data)
    t3 = time.time()
    res = KM._step_E(data)
    t4 = time.time()
    KM._step_M(data,res)
    t5 = time.time()
    KM.distortion(data,res)
    t6 = time.time()
    assignements = np.zeros((n_points,k))
    
    M = dist_matrix(data,KM.means)
    for i in range(n_points):
        index_min = np.argmin(M[i]) #the cluster number of the ith point is index_min
        if (isinstance(index_min,np.int64)):
            assignements[i][index_min] = 1
        else: #Happens when two points are equally distant from a cluster mean
            assignements[i][index_min[0]] = 1
    t7 = time.time()
    
#    print('kmeans :',t1-t0)
#    print('plus_plus :',t2-t1)
#    print('AF_KMC :',t3-t2)
    print('step E :',t4-t3)
    print('step M :',t5-t4)
#    print('distortion :',t6-t5)
#    print('part step E :',t7-t6)
#    center_shift = np.sqrt(np.sum((KM.means_mem[-2] - KM.means_mem[-1]) ** 2, axis=1))
#    center_shift_total = np.sum(center_shift) ** 2
#    print(tol, center_shift_total, KM.iter)