# -*- coding: utf-8 -*-
#
#Created on Fri Apr 21 11:13:09 2017
#
#author: Elina THIBEAU-SUTRE
#
import numpy as np
import scipy.linalg
from scipy.special import gammaln,iv
import os
import warnings
import time


def _full_covariance_matrices(points,means,weights,resp,reg_covar,n_jobs=1):
    """
    Compute the full covariance matrices
    """
    n_components,dim = means.shape
    
    covariance = np.empty((n_components,dim,dim))
    
    for i in range(n_components):
        
        diff = points - means[i]
        diff_weighted = diff * resp[:,i:i+1]
        cov = 1/weights[i] * np.dot(diff_weighted.T,diff)
        cov.flat[::dim + 1] += reg_covar
        covariance[i] = cov
    
    return covariance


def _spherical_covariance_matrices(points,means,weights,assignements,reg_covar):
    """
    Compute the coefficients for the spherical covariances matrices
    """
    n_points,dim = points.shape
    n_components = len(means)
    
    covariance = np.zeros(n_components)

    for i in range(n_components):
        points_centered = points - means[i]
        points_centered_weighted = points_centered * assignements[:,i:i+1]
        product = points_centered * points_centered_weighted
        covariance[i] = np.sum(product)/weights[i]
        covariance[i] += reg_covar
    
    return covariance / dim


def _log_normal_matrix(points,means,cov_chol,covariance_type,n_jobs=1):
    """
    This method computes the log of the density of probability of a normal law centered. Each line
    corresponds to a point from points.
    
    :param points: an array of points (n_points,dim)
    :param means: an array of k points which are the means of the clusters (n_components,dim)
    :param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    :return: an array containing the log of density of probability of a normal law centered (n_points,n_components)
    """
    n_points,_ = points.shape
    n_components,dim = means.shape
    
    if covariance_type == "full":
        
        log_prob = np.empty((n_points,n_components))
        log_det_chol = np.empty(n_components)
        for i in range(n_components):
            precision_chol,_ = scipy.linalg.lapack.dtrtri(cov_chol[i],lower=True)
            y = np.dot(points,precision_chol.T) - np.dot(means[i],precision_chol.T)
            log_prob[:,i] = np.sum(np.square(y),axis=1)
            log_det_chol[i] = np.sum(np.log(np.diagonal(precision_chol)))
        
    elif covariance_type == "spherical":
        precisions_chol = np.reciprocal(cov_chol)
        log_det_chol = dim * np.log(precisions_chol)
        
        log_prob = np.empty(n_components)
        for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):
            y = prec_chol * (points - mu)
            log_prob[:,k] = np.sum(np.square(y))
            
    return -.5 * (dim * np.log(2*np.pi) + log_prob) + log_det_chol

def _log_vMF_matrix(points,means,K,n_jobs=1):
    """
    This method computes the log of the density of probability of a von Mises Fischer law. Each line
    corresponds to a point from points.
    
    :param points: an array of points (n_points,dim)
    :param means: an array of k points which are the means of the clusters (n_components,dim)
    :param cov: an array of k arrays which are the covariance matrices (n_components,dim,dim)
    :return: an array containing the log of density of probability of a von Mises Fischer law (n_points,n_components)
    """
    n_points,dim = points.shape
    n_components,_ = means.shape
    dim = float(dim)
    
    log_prob = K * np.dot(points,means.T)
    # Regularisation to avoid infinte terms
    bessel_term = iv(dim*0.5-1,K)
    idx = np.where(bessel_term==np.inf)[0]
    bessel_term[idx] = np.finfo(K.dtype).max
               
    log_C = -.5 * dim * np.log(2*np.pi) - np.log(bessel_term) + (dim/2-1) * np.log(K)
            
    return log_C + log_prob


def _log_B(W,nu):
    """
    The log of a coefficient involved in the Wishart distribution
    see Bishop book p.693 (B.78)
    """
    
    dim,_ = W.shape
    
    det_W = np.linalg.det(W)
    log_gamma_sum = np.sum(gammaln(.5 * (nu - np.arange(dim)[:, np.newaxis])), 0)
    result = - nu*0.5*np.log(det_W) - nu*dim*0.5*np.log(2)
    result += -dim*(dim-1)*0.25*np.log(np.pi) - log_gamma_sum
    return result


def _log_C(alpha):
    """
    The log of a coefficient involved in the Dirichlet distribution
    see Bishop book p.687 (B.23)
    """
    
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

class BaseMixture():
    """
    Base class for mixture models.
    This abstract class specifies an interface for other mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components=1,init="GMM",n_iter_max=1000,
                 tol=1e-3,patience=0,type_init='resp'):
        
        super(BaseMixture, self).__init__()

        self.n_components = n_components
        self.init = init
        self.type_init = type_init
        
        self._is_initialized = False
        
    def _check_common_parameters(self):
        """
        This function tests the parameters common to all algorithms
        
        """
        
        if self.n_components < 1:
            raise ValueError("The number of components cannot be less than 1")
        else:
            self.n_components = int(self.n_components)
            
            
    def _check_prior_parameters(self,points):
        """
        This function tests the hyperparameters of the VBGMM and the DBGMM
        
        """
        n_points,dim = points.shape
        
        #Checking alpha_0
        if self.alpha_0 is None:
            self.alpha_0 = 1/self.n_components
        elif self.alpha_0 < 0:
            raise ValueError("alpha_0 must be positive")
        
        #Checking beta_0
        if self.beta_0 is None:
            self.beta_0 = 1.0
        
        #Checking nu_0
        if self.nu_0 is None:
            self.nu_0 = dim
        
        elif self.nu_0 < dim:
            raise ValueError("nu_0 must be more than the dimension of the"
                             "problem or the gamma function won't be defined")
        
        
        #Checking prior mean
        if self._means_prior is None:
            self._means_prior = np.mean(points,axis=0)
        elif len(self._means_prior) != dim:
            raise ValueError("the mean prior must have the same dimension as "
                             "the points : %s."
                             % dim)
        
        # Checking prior W-1
        if self.covariance_type == 'full':
            if self._inv_prec_prior is None:
                self._inv_prec_prior = np.cov(points.T)
            elif self._inv_prec_prior.shape != (dim,dim):
                raise ValueError("the covariance prior must have the same "
                                 "dimension as the points : %s."
                                 % dim)
                
        elif self.covariance_type == 'spherical':
            if self._inv_prec_prior is None:
                self._inv_prec_prior = np.var(points)
            elif not isinstance(self._inv_prec_prior, float):
                raise ValueError("Please enter a float for "
                                 "the spherical covariance prior.")
        
    def _check_points(self,points):
        """
        This method checks that the points have the same dimension than the
        problem
        
        """
        
        
        if len(points.shape) == 1:
            points=points.reshape(1,len(points))
            
        elif len(points.shape) != 2:
            raise ValueError('Only 2D or 1D arrays are admitted')

        _,dim_points = points.shape
        _,dim_means = self.means.shape
        if dim_means != dim_points:
            raise ValueError('The points given must have the same '
                             'dimension as the problem : ' + str(dim_means))
        return points
    
    
    def fit(self,points,patience=None,directory=None,saving=None,legend="",
            verbose=False):
        """The EM algorithm
        
        Parameters
        ----------
        points_data : array (n_points,dim)
            A 2D array of points on which the model will be trained
            
        tol : float, defaults to 1e-3
            The EM algorithm will stop when the difference between two steps 
            regarding the convergence criterion is less than tol.
        
        Other Parameters
        ----------------
        points_test : array (n_points_bis,dim) | Optional
            A 2D array of points on which the model will be tested.
            
        saving : str | Optional
            Allows the user to save the model parameters in the directory given
            by the user. Options are ['log','final'].
            
        directory : str | Optional
            Give the emplacement where data of the model will be saved if saving
            is not None.
            
        legend : str | Optional
            A string added to the name of the hdf5 file which will be saved.
            
        Returns
        -------
        None
        
        """
        
        if not self._is_initialized:
            raise ValueError('The system has to be initialized.')
        
        if directory is None:
            directory = os.getcwd()
        
        n_points,dim = points.shape
        
        for i in range(int(n_points/self.window)):
#        for i in range(n_points):
            point = points[i*self.window:(i+1)*self.window:]
            _,log_resp = self._step_E(point)
            self._sufficient_statistics(point,log_resp)
            self._step_M(log_resp)
            self.iter += self.window
            
            if verbose and self.iter%1 == 0:
                print(self.iter)
                self.create_graph(points[:i*self.window:],legend=str(self.iter))
    
    def predict_log_resp(self,points):
        """
        This function returns the logarithm of each point's responsibilities
        
        Parameters
        ----------
        points : array (n_points_bis,dim)
            a 1D or 2D array of points with the same dimension as the problem
            
        Returns
        -------
        log_resp : array (n_points_bis,n_components)
            the logarithm of the responsibilities
            
        """
        
        points = self._check_points(points)
        
        if self._is_initialized:
            _,log_resp = self._step_E(points)
            return log_resp
    
        else:
            raise Exception("The model is not initialized")
    
    def score(self,points):
        """
        This function return the score of the function, which is the logarithm of
        the likelihood for GMM and the logarithm of the lower bound of the likelihood
        for VBGMM and DPGMM
        
        Parameters
        ----------
        points : array (n_points_bis,dim)
            a 1D or 2D array of points with the same dimension as the problem
            
        Returns
        -------
        score : float
            
        """
        points = self._check_points(points)
            
        if self._is_initialized:
            log_prob,log_resp = self._step_E(points)
            score = self._convergence_criterion(points,log_resp,log_prob)
            return score
        
        else:
            raise Exception("The model is not fitted")
    
    def write(self,group):
        """
        A method creating datasets in a group of an hdf5 file in order to save
        the model
        
        Parameters
        ----------
        group : HDF5 group
            A group of a hdf5 file in reading mode

        """
        group.create_dataset('log_weights',self.log_weights.shape,dtype='float64')
        group['log_weights'][...] = self.log_weights
        group.create_dataset('means',self.means.shape,dtype='float64')
        group['means'][...] = self.means
        group.create_dataset('cov',self.cov.shape,dtype='float64')
        group['cov'][...] = self.cov
        group.attrs['iter'] = self.iter
        group.attrs['time'] = time.time()
        
        if self.name in ['VBGMM','DPGMM']:
            initial_parameters = np.asarray([self.alpha_0,self.beta_0,self.nu_0])
            group.create_dataset('initial parameters',initial_parameters.shape,dtype='float64')
            group['initial parameters'][...] = initial_parameters
            group.create_dataset('means prior',self._means_prior.shape,dtype='float64')
            group['means prior'][...] = self._means_prior
            group.create_dataset('inv prec prior',self._inv_prec_prior.shape,dtype='float64')
            group['inv prec prior'][...] = self._inv_prec_prior

    
    def read_and_init(self,group,points):
        """
        A method reading a group of an hdf5 file to initialize DPGMM
        
        Parameters
        ----------
        group : HDF5 group
            A group of a hdf5 file in reading mode
            
        """
        self.log_weights = np.asarray(group['log_weights'].value)
        self.means = np.asarray(group['means'].value)
        self.cov = np.asarray(group['cov'].value)
        self.iter = group.attrs['iter']
        
        n_components = len(self.means)
        if n_components != self.n_components:
            warnings.warn('You are now currently working with %s components.'
                          % n_components)
            self.n_components = n_components
        
        self.init = 'read_and_init'
        print('I will initialize')
        self.initialize(points)
        print(self._is_initialized)
        
#        if self.name in ['VBGMM','DPGMM']:
#            try:
#                initial_parameters = group['initial parameters'].value
#                self.alpha_0 = initial_parameters[0]
#                self.beta_0 = initial_parameters[1]
#                self.nu_0 = initial_parameters[2]
#                self._means_prior = np.asarray(group['means prior'].value)
#                self._inv_prec_prior = np.asarray(group['inv prec prior'].value)
#            except KeyError:
#                warnings.warn('You are reading a model with no prior '
#                              'parameters. They will be initialized '
#                              'if not already given during __init__')