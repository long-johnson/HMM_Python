# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import copy

class DHMM:
    
    """Implementation of discrete observations Hidden Markov Model.
       Observations are integers from 0 to M-1
    """
    
    def __init__(self, n, m, pi=None, a=None, b=None, seed=None):
        """ Create a Hidden Markov Model by providing its parameters.
        n -- number of states;
        m -- number of symbols in alphabet;
        pi (n) -- initial state distribution vector;
        a (n x n) -- transtition probabilities matrix;
        b (n x m) -- emission probabilities in states;
        seed -- seed if HMM needs to be generated randomly (not evenly)
        """
        if seed is not None:
            np.random.seed(seed)        
        self._n = n
        self._m = m
        # if parameters are not defined - give them some statistically correct
        # default values
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
        # emission matrix
        if b is not None:
            self._b = np.array(b)
        elif seed is None:
            self._b = np.full((n, m), 1.0/m)
        else:
            self._b = np.empty(shape=(n, m))
            for i in range(n):
                self._b[i, :] = _generate_discrete_distribution(m)
    
    def generate_sequence(self, T, seed=None):
        """
        Generate sequence of observations produced by this model
        
        Parameters
        ----------
        T : int
            Length of sequence
        seed : int, optional
            Seed for random generator
            
        Returns
        -------
        return : ndarray
            Generated sequence
        """
        # preparation
        # randomize with accordance to seed
        np.random.seed(seed)
        # prepare array for sequence
        seq = np.empty(T, dtype=int)
        states = np.empty(T, dtype=int)
        # generation
        state = _get_sample_discrete_distr(self._pi)
        states[0] = state
        seq[0] = _get_sample_discrete_distr(self._b[state,:])
        for t in range(1, T):
            state = _get_sample_discrete_distr(self._a[state,:])
            states[t] = state
            seq[t] = _get_sample_discrete_distr(self._b[state,:])
        return seq, states
    
    def _calc_forward_noscale(self, seq):
        """ calculate forward variables (no scaling)
        seq -- sequence of observations (1d array)
        return(1) -- likelihood of sequence being produced by model
        return(2) -- alpha: array of forward variables
        """        
        T = seq.size
        # initialization step:
        alpha = np.empty((T, self._n))
        alpha[0,:] = self._pi[:] * self._b[:,seq[0]]
        # induction step:
        for t in range(T-1):
            for i in range(self._n):
                alpha[t+1,i] = \
                    self._b[i,seq[t+1]] * np.sum(alpha[t,:]*self._a[:,i])
        # termination:
        likelihood = np.sum(alpha[-1,:])
        return likelihood, alpha[:,:]
    
    def _calc_forward_logsumexp(self, seq):
        # TODO: needs to be thought through more carefully
        """ calculate forward variables (log-sum-exp trick)
        seq -- sequence of observations (1d array)
        return(1) -- likelihood of sequence being produced by model
        return(2) -- alpha: array of forward variables
        """        
        T = seq.size
        # initialization step:
        log_alpha = np.empty((T, self._n))
        log_alpha[0, :] = np.log(self._pi[:] * self._b[:,seq[0]])
        # induction step:
        for t in range(T-1):
            # for each state calc forward variable
            for i in range(self._n):
                # calc values under exponent
                log_temps = np.log(self._b[:,seq[t+1]]) + \
                    np.log(self._a[:,i]) + log_alpha[t,:]
                max_log_temp = np.max(log_temps)
                # apply log-sum-exp trick
                log_alpha[t+1,i] = max_log_temp + \
                    np.log(np.sum(np.exp(log_temps[:]-max_log_temp)))
        # termination: apply exp() since we calculated logarithms
        alpha = np.exp(log_alpha[:,:])
        likelihood = np.sum(alpha[T-1,:])
        return likelihood, alpha
    
    def _calc_forward_scaled(self, seq):
        """ calculate forward variables (scaled)
        seq -- sequence of observations, array(T)
        return(1) -- loglikelihood of sequence being produced by model
        return(2) -- sc_alpha: array(T, n) of scaled forward variables
        return(3) -- c: array(T) of scaling coefficients
        """
        T = seq.size
        sc_alpha = np.empty(shape=(T, self._n))
        alpha_pr = np.empty(self._n) # from previous step
        alpha = np.empty(self._n)
        c = np.empty(T)
        # initialization
        alpha_pr[:] = self._pi[:] * self._b[:,seq[0]]
        c[0] = 1.0 / np.sum(alpha_pr[:])
        sc_alpha[0,:] = alpha_pr[:] * c[0]
        # induction
        for t in range(T-1):
            for i in range(self._n):
                alpha[i] = \
                    self._b[i,seq[t+1]] * np.sum(sc_alpha[t,:]*self._a[:,i])
            c[t+1] = 1.0 / np.sum(alpha[:])
            sc_alpha[t+1,:] = c[t+1] * alpha[:]
            alpha_pr = np.array(alpha)
        # termination:
        loglikelihood = -np.sum(np.log(c[:]))
        return loglikelihood, sc_alpha[:,:], c[:]
    
    def _calc_backward_noscale(self, seq):
        """ Calc backward variables given the model and sequence
        seq -- sequence of observations, array(T)
        return -- beta: array(T, n) of backward variables
        """
        T = seq.size
        beta = np.empty(shape=(T, self._n))
        # initialization
        beta[-1, :] = 1.0
        # induction
        for t in reversed(range(T-1)):
            for i in range(self._n):
                beta[t,i] = \
                    np.sum(self._a[i,:] * self._b[:,seq[t+1]] * beta[t+1,:])
        # TODO: return also the likelihood  
        #print "beta" + str(np.sum(beta[0,:]*self._b[:,seq[0]]))
        return beta[:,:]
        
    def _calc_backward_scaled(self, seq, c):
        """ Calc backward variables using standard scaling procedure
        seq -- sequence of observations, array(T)
        c -- array(T) of scaling coefficients
        return -- sc_beta: array(T, n) of scaled backward variables
        """
        T = seq.size
        sc_beta = np.empty(shape=(T, self._n))
        beta_pr = np.empty(self._n) # from previous step
        beta = np.empty(self._n)
        # initialization
        beta_pr[:] = 1.0
        sc_beta[-1, :] = c[-1] * beta_pr[:]
        # induction
        for t in reversed(range(T-1)):
            for i in range(self._n):
                beta[i] = \
                    np.sum(self._a[i,:] * self._b[:,seq[t+1]]  * sc_beta[t+1,:])
            sc_beta[t, :] = c[t] * beta[:]
            beta_pr = np.array(beta)
        # TODO: return also the likelihood
        return sc_beta
    
    def _calc_xi_noscale(self, seq, alpha, beta, p):
        """ Calc xi(t,i,j), t=1..T, i,j=1..N - array of probabilities of
        being in state i and go to state j in time t given the model and seq
        seq -- sequence of observations, array(T)
        alpha -- forward variables, array(T,i)
        beta -- backward variables, array(T,i)
        p -- likelihood of seq being produced by this model
        return - xi, array(T,N,N)
        """
        T = seq.size
        xi = np.empty(shape=(T-1, self._n, self._n))
        for t in range(T-1):
            for i in range(self._n):
                xi[t,i,:] = \
                    alpha[t,i] * self._a[i,:] * self._b[:,seq[t+1]] * beta[t+1,:]
        xi[:,:,:] /= p
        return xi
    
    def _calc_gamma_noscale(self, alpha, beta, p, xi=None):
        """ Calc gamma(t,i), t=1..T, i=1..N -- array of probabilities of
        being in state i at the time t given the model and sequence
        mode 1:
        alpha -- forward variables
        beta -- backward variables
        p -- likelihood of sequence being produced by this model
        mode 2:
        xi -- array of xi values (refer to calc_xi_noscaling)
        """
        T = alpha.shape[0]
        gamma = np.empty(shape=(T,self._n))
        if xi is not None:
            gamma[:-1, :] = np.sum(xi[:,:,:], axis=2)
            gamma[-1, :] = alpha[-1, :] * beta[-1, :] / p
        else:
            gamma[:, :] = alpha[:, :] * beta[:, :] / p
        return gamma[:, :]
        
    def train_baumwelch_noscale(self, seq, rtol, max_iter):
        """ Train a HMM given a training sequence & an initial approximation 
        initial approximation is taken from class parameters
        seq -- training sequence
        rtol -- relative tolerance for termination of iteration process
        max_iter -- max number of iterations
        """
        T = seq.size
        iteration = 0
        p_prev = -100.0 # likelihood on previous iteration
        p = 100.0       # likelihood on cur iteration
        # TODO: if next likelihood is lower than previous - 
        # TODO: then take previous parameters ?
        while np.abs(p_prev-p)/p > rtol and iteration < max_iter:
            p_prev = p
            p, alpha = self._calc_forward_noscaling(seq)
            beta = self._calc_backward_noscaling(seq)
            xi = self._calc_xi_noscaling(seq, alpha, beta, p)
            gamma = self._calc_gamma_noscaling(alpha, beta, p, xi)
            # re-estimation
            self._pi[:] = gamma[0,:]
            for i in range(self._n):
                for j in range(self._n):
                    self._a[i,j] = np.sum(xi[:,i,j]) / np.sum(gamma[:-1,i]) 
            for i in range(self._n):
                for m in range(self._m):
                    gamma_sum = 0.0
                    for t in range(T):
                        if seq[t] == m:
                            gamma_sum += gamma[t,i]
                    self._b[i,m] = gamma_sum / np.sum(gamma[:,i])
            iteration += 1
        likelihood, _ = self._calc_forward_noscaling(seq)
        return likelihood, iteration
    
def choose_best_hmm_using_bauwelch(seq, hmms0_size, n, m, isScale = False,
                                   hmms0=None, rtol=0.1, max_iter=10,
                                   verbose=False):
    """ Train several hmms using baumwelch algorithm and choose the best one
    seq -- training sequence 
    hmms0_size -- number of initial approximations
    n -- number of HMM states
    m -- number of HMM symbols
    isScale -- is scaling needed
    mode1: hmms0 -- array of initial approximations
    mode2: hmms0 -- None -- will be generated randomly
    rtol -- relative tolerance (stopping criterion)
    max_iter -- (stopping criterion)
    """
    # generate approximations if not given any
    if hmms0 is None:
        hmms0 = []
        # seeds for hmm generation from 1 to maximum int32
        seeds = \
            np.random.randint(1, high=np.iinfo(np.int32).max, size=hmms0_size)
        for seed in seeds:
            hmms0.append(DHMM(n, m, seed=seed))
    # calc and choose the best hmm estimate
    p_max = np.finfo(np.float64).min # minimal value possible
    for hmm0 in hmms0:
        # TODO: scaled baum and ternary operator
        if not isScale:
            p, iteration = hmm0.train_baumwelch_noscale(seq, rtol, max_iter)
        else:
            raise NotImplementedError, "Scaled baum-welch is not impl. yet"
        if (p_max < p):
            hmm_best = copy.deepcopy(hmm0)
            p_max = p
        if verbose:
            print "another approximation: p=" + str(p)
            print hmm0.pi
            print hmm0.a
            print hmm0.b
    return hmm_best

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

def estimate_hmm_params_by_seq_and_states(N, M, seq, states):
    """ to check that sequence agrees with hmm produced it
    n -- number of hidden states
    m -- number of symbols in alphabet
    seq -- generated sequence
    states -- hidden states appeared during generation
    """
    T = seq.size
    pi = np.zeros(N)
    a = np.zeros(shape=(N,N))
    b = np.zeros(shape=(N,M))
    pi[states[0]] = 1.0
    for t in range(T-1):
        a[states[t], states[t+1]] += 1.0
    a = np.transpose(np.transpose(a) / np.sum(a, axis=1))
    for t in range(T):
        b[states[t], seq[t]] += 1.0
    b = np.transpose(np.transpose(b) / np.sum(b, axis=1))
    return pi, a, b