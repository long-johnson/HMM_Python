# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import copy

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
    
    def calc_forward_noscaling(self, seq):
        """ calculate forward variables (no scaling)
        seq -- sequence of observations (1d array)
        T -- length of sequence
        return(1) -- likelihood of sequence being produced by model
        return(2) -- alpha: array of forward variables
        """        
        T = seq.size
        # initialization step:
        alpha = np.empty((T, self.n))
        alpha[0,:] = self.pi[:] * self.b[:,seq[0]]
        # induction step:
        for t in range(T-1):
            for i in range(self.n):
                alpha[t+1,i] = \
                    self.b[i,seq[t+1]] * np.sum(alpha[t,:]*self.a[:,i])
        # termination:
        likelihood = np.sum(alpha[-1,:])
        return likelihood, alpha[:,:]
    
    def calc_forward_logsumexp(self, seq, T):
        # TODO: needs to be thought through more carefully
        """ calculate forward variables (log-sum-exp trick)
        seq -- sequence of observations (1d array)
        T -- length of sequence
        return(1) -- likelihood of sequence being produced by model
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
        """ calculate forward variables (scaled)
        seq -- sequence of observations, array(T)
        T -- length of sequence, int
        return(1) -- loglikelihood of sequence being produced by model
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
        beta[-1, :] = 1.0
        # induction
        for t in reversed(range(T-1)):
            for i in range(self.n):
                beta[t,i] = \
                    np.sum(self.a[i,:] * self.b[:,seq[t+1]] * beta[t+1,:])
        # TODO: return also the likelihood        
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
                    np.sum(self.a[i,:] * self.b[:,seq[t+1]]  * sc_beta[t+1,:])
            sc_beta[t, :] = c[t] * beta[:]
            beta_pr = beta.copy()
        # TODO: return also the likelihood
        return sc_beta
    
    def calc_xi_noscaling(self, seq, alpha, beta, p):
        """ Calc xi(t,i,j), t=1..T, i,j=1..N - array of probabilities of
        being in state i and go to state j in time t given the model and seq
        seq -- sequence of observations, array(T)
        alpha -- forward variables, array(T,i)
        beta -- backward variables, array(T,i)
        p -- likelihood of seq being produced by this model
        return - xi, array(T,N,N)
        """
        T = seq.size
        xi = np.empty(shape=(T-1, self.n, self.n))
        for t in range(T-1):
            for i in range(self.n):
                xi[t,i,:] = \
                    alpha[t,i] * self.a[i,:] * self.b[:,seq[t+1]] * beta[t+1,:]
        xi[:,:,:] /= p
        return xi
    
    def calc_gamma_noscaling(self, T, alpha, beta, p, xi=None):
        """ Calc gamma(t,i), t=1..T, i=1..N -- array of probabilities of
        being in state i at the time t given the model and sequence
        T -- length of sequence
        mode 1:
        alpha -- forward variables
        beta -- backward variables
        p -- likelihood of sequence being produced by this model
        mode 2:
        xi -- array of xi values (refer to calc_xi_noscaling)
        """
        gamma = np.empty(shape=(T,self.n))
        if xi is not None:
            gamma[:-1, :] = np.sum(xi[:,:,:], axis=2)
            gamma[-1, :] = alpha[-1, :] * beta[-1, :] / p
        else:
            gamma[:, :] = alpha[:, :] * beta[:, :] / p
        return gamma[:, :]
    
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
        
def train_hmm_baumwelch_noscaling(seq, hmm0, rtol=0.1, max_iter=100):
    """ Train a HMM given a training sequence & an initial approximation 
    seq -- training sequence
    hmm0 -- initial approximation of hmm parameters
    rtol -- relative tolerance for termination of iteration process
    max_iter -- max number of iterations
    """
    T = seq.size
    hmm = copy.deepcopy(hmm0)
    iteration = 0
    p_prev = -100.0 # likelihood on previous iteration (dummy value)
    p = 100.0       # likelihood on cur iteration (dummy value)
    #while not np.isclose(p_prev,p,atol=eps) and iteration < max_iter:
    while np.abs(p_prev-p)/p > rtol and iteration < max_iter:
        print np.abs(p_prev-p)/p
        p_prev = p
        p, alpha = hmm.calc_forward_noscaling(seq)
        beta = hmm.calc_backward_noscaling(seq, T)
        xi = hmm.calc_xi_noscaling(seq, alpha, beta, p)
        gamma = hmm.calc_gamma_noscaling(T, alpha, beta, p)
        # re-estimation
        hmm.pi[:] = gamma[0,:]
        for i in range(hmm.n):
            for j in range(hmm.n):
                hmm.a[i,j] = np.sum(xi[:,i,j]) / np.sum(gamma[:-1,i]) 
        for i in range(hmm.n):
            for m in range(hmm.m):
                gamma_sum = 0
                for t in range(T):
                    if seq[t] == m:
                        gamma_sum += gamma[t,i]
                hmm.b[i,m] = gamma_sum / np.sum(gamma[:,i])
        iteration += 1
    print "train_hmm_baumwelch_noscaling: iteration = " + str(iteration)
    return hmm

def generate_discrete_distribution(n):
    """ Generate n values > 0.0, whose sum equals 1.0
    """
    xs = np.array(stats.uniform.rvs(size=n))
    return xs / np.sum(xs)