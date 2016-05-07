# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import copy
import StandardImputationMethods as imp

class GHMM:
    """Implementation of Hidden Markov Model where observation density is
    represented by a mixture of normal distributions
       Observations are vectors of real numbers
    """
    
    def __init__(self, n, m, z, mu, sig, pi=None, a=None, tau=None, seed=None):
        """ 
        n - number of hidden states
        m - number of distribution mixture components
        z - dimension of observations
        pi - initial state distribution vector (n)
        a - transition probabilities matrix (n x n)
        tau - weights of mixture distributions (n x m)
        mu - means of normal distributions (n x m x z)
        sig - covariations of normal distributions (diagonal elements) (n x m x z)
        
        seed - provide seed if HMM needs to be generated randomly (not evenly)
        """
        self._n = n
        self._m = m
        self._z = z
        # if parameters are not defined - give them some statistically correct
        # default values or generated randomly if seed is provided
        if pi is not None:
            self._pi = np.array(pi)
        elif seed is None:
            self._pi = np.full(n, 1.0/n)
        else:
            self._pi = _generate_discrete_distribution(n)
        # transition matrix
        if a is not None:
            self._a = np.array(a)
        elif seed is None:
            self._a = np.full((n, n), 1.0/n)
        else:
            self._a = np.empty(shape=(n, n))
            for i in range(n):
                self._a[i, :] = _generate_discrete_distribution(n)      
        # mixture weights
        if tau is not None:
            self._tau = tau
        elif seed is None:
            self._tau = np.full((n,m), 1.0/m)
        else:
            self._tau = np.empty(shape=(n, m))
            for i in range(n):
                self._tau[i,:] = _generate_discrete_distribution(m)
        # TODO: add random generation of mu and sig
        self._mu = mu
        self._sig = sig
        
    def generate_sequences(self, K, T, seed=None):
        """
        Generate sequences of observations produced by this model
        
        Parameters
        ----------
        K : int
            number of sequences
        T : int
            Length of each sequence
        seed : int, optional
            Seed for random generator
            
        Returns
        -------
        seqs : list of 1darrays
            List of generated sequences
        state_seqs : list of 1darrays
            List of hidden states sequences used for generation
        """
        # preparation
        # randomize with accordance to seed
        np.random.seed(seed)
        # prepare list for sequences
        seqs = [np.empty((T,self._z),dtype=np.float64) for k in range(K)]
        state_seqs = [np.empty(T, dtype=np.int32) for k in range(K)]
        # generation
        for k in range(K):
            state = _get_sample_discrete_distr(self._pi)
            state_seqs[k][0] = state
            mix_elem = _get_sample_discrete_distr(self._tau[state,:])
            cov_matrix = np.diag(self._sig[state, mix_elem])
            seqs[k][0] = \
                np.random.multivariate_normal(self._mu[state, mix_elem],
                                              cov_matrix)
            for t in range(1, T):
                state = _get_sample_discrete_distr(self._a[state,:])
                state_seqs[k][t] = state
                mix_elem = _get_sample_discrete_distr(self._tau[state,:])
                cov_matrix = np.diag(self._sig[state, mix_elem])
                seqs[k][t] = \
                    np.random.multivariate_normal(self._mu[state, mix_elem],
                                                  cov_matrix)
        return seqs, state_seqs
        
def _generate_discrete_distribution(n):
    """ Generate n values > 0.0, whose sum equals 1.0
    """
    xs = np.array(stats.uniform.rvs(size=n))
    return xs / np.sum(xs)
    
def _get_sample_discrete_distr(distr):
    """ Get sample of random value that has discrete distrubution 'distr'
        random values are 0, 1, ...
    """
    val = np.random.uniform()
    cum = 0.0
    for i in range(len(distr)):
        cum += distr[i]
        if val < cum:
            return i
    return distr.size-1
