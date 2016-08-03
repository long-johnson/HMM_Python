# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
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
    
   
    def _calc_b(self, seq):
        """
        Calc conditional densities of each sequence element given each HMM state
        
        Parameters
        ----------
        seq : 2darray (T x Z)
            observations sequence 
        
        Returns
        -------
        b : 2darray (T x N)
            conditional densities for each sequence element and HMM state
        """
        N = self._n
        M = self._m
        T = seq.shape[0]       
        mu = self._mu
        sig = self._sig
        tau = self._tau
        b = np.empty(shape=(T, N))
        # TODO: optimize, remove inner loop
        # TODO: optimize, move pdf calculations to other routine
        for t in range(T):
            for i in range(self._n):
                b[t, i] = sum([tau[i,m] * \
                    sp.stats.multivariate_normal.pdf(seq[t], mu[i,m], sig[i,m])\
                    for m in range(M)])    
        return b
    
    def _calc_forward_scaled(self, seq, b):
        """
        Calc scaled forward variables
        
        Parameters
        ----------
        seq : 2darray (T x Z)
            observations sequence
        b : 2darray (T x N)
            conditional densities for each sequence element and HMM state
        
        Returns
        ----------
        likelihood : float64
            likelihood of the sequence being generated by the current HMM
        alpha : 2darray (T x N)
            scaled forward variables
        c : 2darray (T)
            scale coefficients
        """
        N = self._n
        T = seq.shape[0]       
        pi = self._pi
        a = self._a
        # memory
        alpha = np.empty(shape=(T, N))
        c = np.empty(T)
        # initialization
        alpha_t = pi * b[0]
        c[0] = 1.0 / np.sum(alpha_t)
        alpha[0,:] = c[0] * alpha_t 
        # induction
        for t in range(T-1):
            # TODO: optimize
            for i in range(N):
                alpha_t[i] = b[t+1,i] * np.sum(alpha[t,:]*a[:,i])
            c[t+1] = 1.0 / np.sum(alpha_t)
            alpha[t+1,:] = c[t+1] * alpha_t
        # termination:
        loglikelihood = -np.sum(np.log(c))
        return loglikelihood, alpha, c
    
    def _calc_backward_scaled(self, seq, b, c):
        """
        Calc scaled backward variables
        
        Parameters
        ----------
        seq : 2darray (T x Z)
            observations sequence
        b : 2darray (T x N)
            conditional densities for each sequence element and HMM state
        c : 2darray (T)
            scaling coefficients
        
        Returns
        ----------
        beta : 2darray (T x N)
            scaled backward variables
        """
        N = self._n
        T = seq.shape[0]       
        a = self._a
        # memory
        beta = np.empty(shape=(T, N))
        # initialization
        beta_t = np.full(N, 1.0)
        beta[-1,:] = c[-1] * beta_t
        # induction
        for t in reversed(range(T-1)):
            # TODO: optimize
            for i in range(N):
                beta_t[i] = np.sum(a[i,:] * b[t+1,:]  * beta[t+1,:])
            beta[t,:] = c[t] * beta_t
        return beta
        
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

