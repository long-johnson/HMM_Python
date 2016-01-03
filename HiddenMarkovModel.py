# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

class HiddenMarkovModel:
    
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
        self.n = n
        self.m = m
        # if parameters are not defined - give them some statistically correct
        # default values
        if pi is not None:
            self.pi = pi.copy()
        elif seed is None:
            self.pi = np.full(n, 1.0/n)
        else:
            self.pi = generate_discrete_distribution(n)
        # transition matrix
        if a is not None:
            self.a = a.copy()
        elif seed is None:
            self.a = np.full((n, n), 1.0/n)
        else:
            self.a = np.empty(shape=(n, n))
            for i in range(n):
                self.a[i, :] = generate_discrete_distribution(n)          
        # emission matrix
        if b is not None:
            self.b = b.copy()
        elif seed is None:
            self.b = np.full((n, m), 1.0/m)
        else:
            self.b = np.empty(shape=(n, m))
            for i in range(n):
                self.b[i, :] = generate_discrete_distribution(m)
    
    def calc_forward_noscaling(self, seq, T):
        """ l-hood of sequence being produced by this model (no scaling)
        seq -- sequence of observations (1d array)
        T -- length of sequence
        return(1) -- likelihood
        return(2) -- alpha: array of forward variables
        """        
        # initialization step:
        alpha = np.empty((T, self.n))
        alpha[0,:] = self.pi * self.b[:,seq[0]]
        # induction step:
        for t in range(T-1):
            for i in range(self.n):
                alpha[t+1,i] = \
                    self.b[i,seq[t+1]] * np.sum(alpha[t,:]*self.a[:,i])
        # termination:
        likelihood = np.sum(alpha[T-1,:])
        return likelihood, alpha
    
    def calc_forward_logsumexp(self, seq, T):
        # TODO: needs to be thought through more carefully
        """ l-hood of sequence being produced by model (log-sum-exp trick)
        seq -- sequence of observations (1d array)
        T -- length of sequence
        return(1) -- likelihood
        return(2) -- alpha: array of forward variables
        """        
        # initialization step:
        log_alpha = np.empty((T, self.n))
        log_alpha[0, :] = np.log(self.pi[:] * self.b[:,seq[0]])
        # induction step:
        for t in range(T-1):
            # for each state calc forward variable
            for i in range(self.n):
                # calc values under exponent
                log_temps = np.log(self.b[:,seq[t+1]]) + \
                    np.log(self.a[:,i]) + log_alpha[t,:]
                max_log_temp = np.max(log_temps)
                # apply log-sum-exp trick
                log_alpha[t+1,i] = max_log_temp + \
                    np.log(np.sum(np.exp(log_temps[:]-max_log_temp)))
        # termination: apply exp() since we calculated logarithms
        alpha = np.exp(log_alpha[:,:])
        likelihood = np.sum(alpha[T-1,:])
        return likelihood, alpha
    
    def calc_forward_scaled(self, seq, T):
        """ l-hood of sequence being produced by model (scaled)
        seq -- sequence of observations, array(T)
        T -- length of sequence, int
        return(1) -- loglikelihood
        return(2) -- sc_alpha: array(T, n) of scaled forward variables
        return(3) -- c: array(T) of scaling coefficients
        """
        sc_alpha = np.empty(shape=(T, self.n))
        alpha_pr = np.empty(self.n) # from previous step
        alpha = np.empty(self.n)
        c = np.empty(T)
        # initialization
        alpha_pr[:] = self.pi[:] * self.b[:,seq[0]]
        c[0] = 1.0 / np.sum(alpha_pr[:])
        sc_alpha[0,:] = alpha_pr[:] * c[0]
        # induction
        for t in range(T-1):
            for i in range(self.n):
                alpha[i] = \
                    self.b[i,seq[t+1]] * np.sum(sc_alpha[t,:]*self.a[:,i])
            c[t+1] = 1.0 / np.sum(alpha[:])
            sc_alpha[t+1,:] = c[t+1] * alpha[:]
            alpha_pr = alpha.copy()
        # termination:
        loglikelihood = -np.sum(np.log(c[:]))
        return loglikelihood, sc_alpha[:,:], c[:]
    
    def calc_backward_noscaling(self, seq, T):
        """ Calc backward variables given the model and sequence
        seq -- sequence of observations, array(T)
        T -- length of sequence, int
        return -- beta: array(T, n) of backward variables
        """
        beta = np.empty(shape=(T, self.n))
        # initialization
        beta[T-1, :] = 1.0
        # induction
        for t in reversed(range(T-1)):
            for i in range(self.n):
                beta[t,i] = self.b[i,seq[t+1]] * sum(self.a[i,:] * beta[t+1,:])
        return beta[:,:]
        
    def calc_backward_scaled(self, seq, T, c):
        """ Calc backward variables using standard scaling procedure
        seq -- sequence of observations, array(T)
        T -- length of sequence, int
        c -- array(T) of scaling coefficients
        return -- sc_beta: array(T, n) of scaled backward variables
        """
        sc_beta = np.empty(shape=(T, self.n))
        beta_pr = np.empty(self.n) # from previous step
        beta = np.empty(self.n)
        # initialization
        beta_pr[:] = 1.0
        sc_beta[-1, :] = c[-1] * beta_pr[:]
        # induction
        for t in reversed(range(T-1)):
            for i in range(self.n):
                beta[i] = \
                    self.b[i,seq[t+1]] * sum(self.a[i,:] * sc_beta[t+1,:])
            sc_beta[t, :] = c[t] * beta[:]
            beta_pr = beta.copy()
        return sc_beta
    
    def generate_sequence(self, T, seed=None):
        """ ... of observations produced by this model
        T -- length of sequence
        seed -- seed for random generator
        return -- generated sequence
        """
        # preparation
        # randomize with accordance to seed
        np.random.seed(seed)
        # prepare array for sequence
        seq = np.empty(T, dtype=int)
        # discrete distrubution of initial states
        pi_distr = \
            stats.rv_discrete(values = (np.arange(self.n), self.pi))
        # set dicrete distribution of transtition probabilities for each state
        a_distr_list = []
        for i in range(self.n):
            temp = stats.rv_discrete(values=(np.arange(self.n),(self.a[i,:])))
            a_distr_list.append(temp)
        # set dicrete distribution of symbols for each state
        b_distr_list = []
        for i in range(self.n):
            temp = stats.rv_discrete(values=(np.arange(self.m),(self.b[i,:])))
            b_distr_list.append(temp)
        # generation
        # generate number of initial state
        state = pi_distr.rvs()
        # generate observation from the state
        seq[0] = b_distr_list[state].rvs()
        for t in range(1, T):
            # transit to a new state
            state = a_distr_list[state].rvs()
            # generate observation from the state
            seq[t] = b_distr_list[state].rvs()
        return seq
        
    #def train(self, )

def generate_discrete_distribution(n):
    """ Generate n values > 0.0, whose sum equals 1.0
    """
    xs = np.array(stats.uniform.rvs(size=n))
    return xs / np.sum(xs)