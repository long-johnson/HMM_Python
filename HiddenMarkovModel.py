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
            
        if a is not None:
            self.a = a.copy()
        elif seed is None:
            self.a = np.full((n, n), 1.0/n)
        else:
            self.a = np.empty(shape=(n, n))
            for i in range(n):
                self.a[i, :] = generate_discrete_distribution(n)          
            
        if b is not None:
            self.b = b.copy()
        elif seed is None:
            self.b = np.full((n, m), 1.0/m)
        else:
            self.b = np.empty(shape=(n, m))
            for i in range(n):
                self.b[i, :] = generate_discrete_distribution(m)
    
    def calc_likelihood_noscaling(self, seq, T):
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
            alpha[t+1,:] = \
                self.b[:,seq[t+1]] * np.sum(alpha[t,:]*self.a[:,:], axis=1)
        likelihood = np.sum(alpha[T-1, :])
        return likelihood, alpha
        
    def calc_likelihood_logsumexp(self, seq, T):
        """ l-hood of sequence being produced by model (log-sum-exp trick)
        seq -- sequence of observations (1d array)
        T -- length of sequence
        return(1) -- loglikelihood
        return(2) -- alpha: array of forward variables
        """        
        # initialization step:
        log_alpha = np.empty((T, self.n))
        log_alpha[0,:] = np.log(self.pi * self.b[:,seq[0]])
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
                    np.log(np.sum(np.exp(log_temps-max_log_temp)))
        # apply exp() since we calculated logarithms
        alpha = np.exp(log_alpha[:,:])
        likelihood = np.sum(alpha[T-1,:])
        return likelihood, alpha
        
    
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
    return xs / (np.sum(xs))