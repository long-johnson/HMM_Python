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
        sig - covariation matrix of normal distributions (n x m x z x z)
        
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
            cov_matrix = self._sig[state, mix_elem]
            seqs[k][0] = \
                np.random.multivariate_normal(self._mu[state, mix_elem],
                                              cov_matrix)
            for t in range(1, T):
                state = _get_sample_discrete_distr(self._a[state,:])
                state_seqs[k][t] = state
                mix_elem = _get_sample_discrete_distr(self._tau[state,:])
                cov_matrix = self._sig[state, mix_elem]
                seqs[k][t] = \
                    np.random.multivariate_normal(self._mu[state, mix_elem],
                                                  cov_matrix)
        return seqs, state_seqs
    
    def calc_likelihood(self, seqs):
        """
        Calc likelihood of the sequences being generated by the current HMM
        
        Parameters
        ----------
        seqs : list of 2darrays (T x Z)
            observations sequences
        Returns
        -------
        likelihood : float64
            likelihood of the sequences being generated by the current HMM
        """
        likelihood = 0.0
        for seq in seqs:
            b, _ = self._calc_b(seq)
            likelihood += self._calc_forward_scaled(seq, b)[0]
        return likelihood
   
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
        g : 3darray (T x N x M)
            pdf (Gaussian distribution) values for each sequence element
        """
        N = self._n
        M = self._m
        T = seq.shape[0]
        mu = self._mu
        sig = self._sig
        tau = self._tau
        g = np.empty((T, N, M))
        for t in range(T):
            for i in range(N):
                for m in range(M):
                    g[t, i, m] = sp.stats.multivariate_normal.pdf(seq[t], mu[i,m], sig[i,m])
        b = np.sum(tau * g, axis=2)
        return b, g
    
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
        # memory
        alpha = np.empty(shape=(T, N))
        c = np.empty(T)
        # initialization
        alpha_t = pi * b[0]
        c[0] = 1.0 / np.sum(alpha_t)
        alpha[0,:] = c[0] * alpha_t 
        # induction
        a_T = np.transpose(self._a)
        for t in range(T-1):
            alpha_t = b[t+1,:] * np.sum(alpha[t,:]*a_T, axis=1)
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
            beta_t = np.sum(a * b[t+1,:]  * beta[t+1,:], axis=1)
            beta[t,:] = c[t] * beta_t
        return beta
        
    def _calc_xi_gamma_scaled(self, seq, b, alpha, beta):
        """ Calc xi(t,i,j), t=1..T, i,j=1..N - array of probabilities of
        being in state i and go to state j in time t given the model and seq
        Calc gamma(t,i), t=1..T, i=1..N -- array of probabilities of
        being in state i at the time t given the model and sequence
        
        Parameters
        ----------
        
        seq : 2darray (TxZ)
            sequence of observations
        b : 2darray (T x N)
            conditional densities for each sequence element and HMM state
        alpha : 2darray (TxN)
            forward variables
        beta : 2darray (TxN)
            backward variables
            
        Returns
        -------
        
        xi : 3darray (TxNxN)
            probs of transition from i to j at time t given the model and seq
        gamma : 2darray (TxN)
            probs of transition from i at time t given the model and seq
        """
        T = seq.shape[0]
        N = self._n
        xi = np.empty(shape=(T-1, N, N))
        a_tr = np.transpose(self._a)
        for t in range(T-1):                  
            xi[t,:,:] = (alpha[t,:] * a_tr).T * b[t+1,:] * beta[t+1,:]
        gamma = np.sum(xi, axis=2)
        return xi, gamma
     
    def _calc_gamma_m_scaled(self, seq, b, g, gamma):
        """ Calc gamma_m(t,i,m), t=1..T, i=1..N, m=1..M -- array of probs
        of being in state i at time t and selecting m-th mixture component
        
        Parameters
        ----------
        seq : 2darray (TxZ)
            sequence of observations
        b : 2darray (T x N)
            conditional densities for each sequence element and HMM state
        g : 3darray (T x N x M)
            pdf (Gaussian distribution) values for each sequence element
        gamma : 2darray (TxN)
            probs of transition from i at time t given the model and seq
            
        Returns
        -------
        gamma_m : 3darray (TxNxM)
            probs of transition from i at time t and selection of m-th mixture
        """
        N = self._n
        M = self._m
        T = seq.shape[0]       
        tau = self._tau
        gamma_m = np.empty(shape=(T-1, N, M))
        gamma_m = g[:-1,:,:] * tau * gamma[:,:, np.newaxis] / b[:-1,:,np.newaxis]
        return gamma_m
    
    def train_baumwelch(self, seqs, rtol, max_iter):
        N = self._n
        M = self._m
        Z = self._z
        K = len(seqs)
        T = max([len(seq) for seq in seqs]) # for gamma_ms
        iteration = 0
        p_prev = -100.0 # likelihood on previous iteration
        p = 100.0       # likelihood on cur iteration
        while np.abs((p_prev-p)/p) > rtol and iteration < max_iter:
            p_prev = p
            # calculate numenator and denominator for re-estimation
            pi_up = np.zeros(N)
            a_up = np.zeros((N, N))
            tau_up = np.zeros((N, M))
            a_tau_down = np.zeros(N)
            mu_up = np.zeros((N, M, Z))
            mu_sig_down = np.zeros((N, M))
            gamma_ms = np.zeros((K,T-1,N,M))
            sig_up = np.zeros((N, M, Z, Z))
            for k in range(K):   
                # expectation
                seq = seqs[k]
                b, g = self._calc_b(seq)
                p, alpha, c = self._calc_forward_scaled(seq, b)
                beta = self._calc_backward_scaled(seq, b, c)
                xi, gamma = self._calc_xi_gamma_scaled(seq, b, alpha, beta)
                gamma_ms[k] = self._calc_gamma_m_scaled(seq, b, g, gamma)
                # accumulating for maximization
                pi_up += gamma[0,:]
                a_up += np.sum(xi, axis=0)
                sum_gamma = np.sum(gamma, axis=0)                
                tau_up += np.sum(gamma_ms[k], axis=0)
                a_tau_down += sum_gamma
                mu_up += np.einsum('tnm,tz->nmz', gamma_ms[k], seq[:-1])
                mu_sig_down += np.sum(gamma_ms[k], axis=0)
            # re-estimation
            self._pi = pi_up / K
            self._a = (a_up.T / a_tau_down).T
            self._tau = (tau_up.T / a_tau_down).T
            self._mu = mu_up / mu_sig_down[:,:,np.newaxis]
            # accumulating sig
            # TODO: is it possible to optimize this ...?
            for k in range(K):
                seq = seqs[k]
                T = seq.shape[0]
                for t in range(T-1):
                    diff = self._mu - seq[t]
                    for i in range(N):
                        for m in range(M):
                            sig_up[i,m] += gamma_ms[k,t,i,m] * \
                                           diff[i,m] * (diff[i,m]).reshape((Z,1))
                                           #instead of np.outer()
            # sig re-estimation
            self._sig = sig_up / mu_sig_down[:,:,np.newaxis,np.newaxis]
            iteration += 1
        likelihood = self.calc_likelihood(seqs)
        return likelihood, iteration

def train_best_hmm_baumwelch(seqs, hmms0_size, N, M, Z, hmms0=None, rtol=1e-1, 
                             max_iter=None, verbose=False):
    """ Train several hmms using baumwelch algorithm and choose the best one
    
    Parameters
    ----------
    seqs : list of float64 2darrays (TxZ)
        list of training sequences
    hmms0_size : int
        number of initial approximations
    N : int
        number of HMM states
    M : int
        number of HMM symbols
    Z : int
        dimensionality of observations
    hmms0 : list of GHMMs, optional
        list of initial approximations of HMM parameters
        !note: len(hmms0) == hmms0_size must be fulfilled!
        if not specified, hmms0_size approximations will be generated
    rtol : float64, optional
        relative tolerance (stopping criterion)
    max_iter : float64, optional
        maximal number of Baum-Welch iterations (stopping criterion)
    verbose : bool, optional
        controls whether some debug info should be printed to stdout
    
    Returns
    -------
    hmm_best : GHMM
        best trained hmm or None
    iter_best : float64
        number of iterations to train the best hmm 
    """
    # TODO: generate approximations if not given any
    # TODO: generate mu and sig according to seqs, but slightly random
    # calc and choose the best hmm estimate
    hmms = copy.deepcopy(hmms0)
    p_max = np.finfo(np.float64).min # minimal value possible
    hmm_best = None
    iter_best = -1 # count number of iters for the best hmm
    for hmm in hmms:
        p, iteration = hmm.train_baumwelch(seqs, rtol, max_iter)
        if (p_max < p and np.isfinite(p)):
            hmm_best = hmm
            p_max = p
            iter_best = iteration
        if verbose:
            print "another approximation: p=" + str(p)
            print "iteration = " + str(iteration)
            print hmm._pi
            print hmm._a
            print hmm._tau
            print hmm._mu
            print hmm._sig
    return hmm_best, iter_best
        
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

