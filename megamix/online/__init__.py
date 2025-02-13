# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:50:46 2017

@author: Elina Thibeau-Sutre
"""
# Marvin need to check if compiled version leads to improvement boost
# try:
#     from .cython_version.GMM_cython import GaussianMixture
#     from .cython_version.kmeans_cython import Kmeans,dist_matrix
#     from .cython_version.base_cython import cholupdate
#     print('compiled')
# except:
#     from .GMM import GaussianMixture
#     from .kmeans import Kmeans,dist_matrix
#     from .base import cholupdate
#     print('pure python mode')

from .GMM import GaussianMixture
from .kmeans import Kmeans,dist_matrix
from .base import cholupdate
print('pure python mode')
from .VBGMM import VariationalGaussianMixture

__all__ = ['GaussianMixture','Kmeans','cholupdate','dist_matrix',
           'VariationalGaussianMixture']