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
        seqs = [np.empty(T,dtype=np.int32) for k in range(K)]
        state_seqs = [np.empty(T,dtype=np.int32) for k in range(K)]
        # generation
        for k in range(K):
            state = _get_sample_discrete_distr(self._pi)
            state_seqs[k][0] = state
            seqs[k][0] = _get_sample_discrete_distr(self._b[state,:])
            for t in range(1, T):
                state = _get_sample_discrete_distr(self._a[state,:])
                state_seqs[k][t] = state
                seqs[k][t] = _get_sample_discrete_distr(self._b[state,:])
        return seqs, state_seqs
    
    def _calc_forward_noscale(self, seq, avail=None):
        """ calculate forward variables (no scaling)
        seq -- sequence of observations (1d array)
        avail -- availability of sequence elements
        return(1) -- likelihood of sequence being produced by model
        return(2) -- alpha: array of forward variables
        """
        # if avail is not defined consider that all observations are available
        if avail is None:
            avail = np.full(seq.size, fill_value=True)
        T = seq.size
        # initialization step:
        alpha = np.empty((T, self._n))
        if avail[0]:
            alpha[0,:] = self._pi[:] * self._b[:,seq[0]]
        else:
            alpha[0,:] = self._pi[:]
        # induction step:
        for t in range(T-1):
            if avail[t+1]:
                for i in range(self._n):
                    alpha[t+1,i] = \
                        self._b[i,seq[t+1]] * np.sum(alpha[t,:]*self._a[:,i])
            else:
                for i in range(self._n):
                    alpha[t+1,i] = np.sum(alpha[t,:] * self._a[:,i])
        # termination:
        likelihood = np.sum(alpha[-1,:])
        return likelihood, alpha[:,:]
    
    def calc_likelihood_noscale(self, seqs, avails=None):
        """ calculate average likelihood that all the sequences
        was produced by this model
        seqs -- list of sequences
        return -- average likelihood
        """
        if avails is None:
            return np.sum([self._calc_forward_noscale(seq)[0] for seq in seqs])
        else:
            return np.sum([self._calc_forward_noscale(seqs[k], avails[k])[0] \
                           for k in range(len(seqs))])
    
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
            # TODO: optimize
            for i in range(self._n):
                alpha[i] = \
                    self._b[i,seq[t+1]] * np.sum(sc_alpha[t,:]*self._a[:,i])
            c[t+1] = 1.0 / np.sum(alpha[:])
            sc_alpha[t+1,:] = c[t+1] * alpha[:]
            alpha_pr = np.array(alpha)
        # termination:
        loglikelihood = -np.sum(np.log(c[:]))
        return loglikelihood, sc_alpha[:,:], c[:]
    
    def _calc_backward_noscale(self, seq, avail=None):
        """ Calc backward variables given the model and sequence
        seq -- sequence of observations, array(T)
        return -- beta: array(T, n) of backward variables
        """
        # if avail is not defined consider that all observations are available
        if avail is None:
            avail = np.full(seq.size, fill_value=True)
        T = seq.size
        beta = np.empty(shape=(T, self._n))
        # initialization
        beta[-1, :] = 1.0
        # induction
        for t in reversed(range(T-1)):
            if avail[t+1]:
                # TODO: optimize
                #beta[t,:] = np.sum(np.prod(self._a[:,:] * self._b[:,seq[t+1]], axis=1) * beta[t+1,:], axis=0)
                for i in range(self._n):
                    beta[t,i] = \
                        np.sum(self._a[i,:] * self._b[:,seq[t+1]] * beta[t+1,:])
            else:
                # TODO: optimize
                for i in range(self._n):
                    beta[t,i] = np.sum(self._a[i,:] * beta[t+1,:])
        # likelihood  
        #print "beta likelihood" + str(np.sum(beta[0,:]*self._b[:,seq[0]]))
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
    
    def _calc_xi_noscale(self, seq, alpha, beta, p, avail=None):
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
        if avail is None:
            for t in range(T-1):
                for i in range(self._n):                    
                    xi[t,i,:] = \
                        alpha[t,i] * self._a[i,:] * self._b[:,seq[t+1]] * beta[t+1,:]
        else:
            for t in range(T-1):
                if avail[t+1]:
                    for i in range(self._n):                    
                        xi[t,i,:] = alpha[t,i] * self._a[i,:] * \
                                    self._b[:,seq[t+1]] * beta[t+1,:]
                else:
                    for i in range(self._n):
                        xi[t,i,:] = alpha[t,i] * self._a[i,:] * beta[t+1,:]
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
        
    def train_baumwelch_noscale(self, seqs, rtol, max_iter, avails = None):
        """ Train a HMM given a training sequence & an initial approximation 
        initial approximation is taken from class parameters
        seqs -- list of K training sequences of various length
        rtol -- relative tolerance for termination of iteration process
        max_iter -- max number of iterations
        """
        K = len(seqs)
        iteration = 0
        p_prev = -100.0 # likelihood on previous iteration
        p = 100.0       # likelihood on cur iteration
        # TODO: if next likelihood is lower than previous - 
        # TODO: then take previous parameters ?
        while np.abs(p_prev-p)/p > rtol and iteration < max_iter:
            p_prev = p
            pi_up = np.zeros(self._n)
            a_up = np.zeros(shape=(self._n, self._n))
            a_down = np.zeros(self._n)
            b_up = np.zeros(shape=(self._n, self._m))
            b_down = np.zeros(shape=(self._n))
            for k in range(K):
                # expectation
                seq = seqs[k]
                avail = avails[k] if avails is not None else None
                p, alpha = self._calc_forward_noscale(seq, avail)
                beta = self._calc_backward_noscale(seq, avail)
                xi = self._calc_xi_noscale(seq, alpha, beta, p, avail)
                gamma = self._calc_gamma_noscale(alpha, beta, p, xi)
                # accumulating for maximization
                pi_up[:] += gamma[0,:]                
                a_up[:,:] += np.sum(xi[:,:,:], axis=0)
                temp = np.sum(gamma[:-1,:], axis=0)
                a_down[:] += temp
                if avails is None:
                    # sum all gammas where observation is symbol m  
                    for m in range(self._m):
                        b_up[:,m] += np.sum(gamma[seq[:]==m,:], axis=0)
                    b_down[:] += temp + gamma[-1,:]
                else:
                    # sum all gammas where observation is avail and is symbol m                  
                    for m in range(self._m):
                        condition = (seq[:]==m) & (avail[:])
                        b_up[:,m] += np.sum(gamma[condition,:], axis=0)
                    # sum all gammas where observation is availiable
                    b_down[:] += np.sum(gamma[avail[:],:], axis=0)
            # re-estimation
            self._pi[:] = pi_up[:] / K
            self._a[:,:] = (a_up[:,:].T / a_down[:]).T
            self._b[:,:] = (b_up[:,:].T / b_down[:]).T
            iteration += 1
        # TODO: what if various length?
        likelihood = self.calc_likelihood_noscale(seqs, avails)
        return likelihood, iteration
        
    def train_baumwelch_gluing(self, seqs, rtol, max_iter, avails,
                              isScale=False):
        """ Glue segments between gaps together and then train Baum-Welch
        """
        K = len(seqs)
        # remove all gaps and just glue remaining segments together
        seqs_glued = []
        for k in range(K):
            glued = seqs[k][avails[k]]  # fancy indexing
            seqs_glued.append(glued)
        if isScale:
            raise NotImplementedError, "Scaled baum-welch is not impl. yet"
        else:
            likelihood, iteration = \
                self.train_baumwelch_noscale(seqs_glued, rtol, max_iter)
        return likelihood, iteration
        
    def train_baumwelch_multiseq(self, seqs, rtol, max_iter, avails,
                                      isScale=False, min_len=1):
        """ Slice sequence with gaps into multisequence and then train
            min_len -- minimal length of sequence in final multisequence
        """
        multiseq = []
        K = len(seqs)
        for k in range(K):
            seq = seqs[k]
            gap_indexes = np.nonzero(avails[k]==False)[0]
            # split seqs by gap_indexes
            seq_splits = np.split(seq, gap_indexes)
            # then trim every first item in every split but not in first split
            if seq_splits[0].size >= min_len:
                multiseq.append(seq_splits[0])
            for seq_split in seq_splits[1:]:
                to_add = seq_split[1:]
                if to_add.size >= min_len:
                    multiseq.append(to_add)
        #print multiseq
        if isScale:
            raise NotImplementedError, "Scaled baum-welch is not impl. yet"
        else:
            likelihood, iteration = \
                self.train_baumwelch_noscale(multiseq, rtol, max_iter)
        return likelihood, iteration
            
    
def choose_best_hmm_using_bauwelch(seqs, hmms0_size, n, m, algorithm='marginalization',
                                   isScale=False, hmms0=None, rtol=1e-1, 
                                   max_iter=None, avails = None, verbose=False):
    """ Train several hmms using baumwelch algorithm and choose the best one
    seqs -- list of training sequences
    hmms0_size -- number of initial approximations
    n -- number of HMM states
    m -- number of HMM symbols
    isScale -- is scaling needed
    mode1: hmms0 -- array of initial approximations
    mode2: hmms0 -- None -- will be generated randomly
    rtol -- relative tolerance (stopping criterion)
    max_iter -- (stopping criterion)
    return: hmm_best, iter_max
    """
    assert algorithm in ('marginalization', 'gluing'),\
        "Invalid algorithm '{}'".format(algorithm)
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
    hmm_best = None
    iter_max = -1
    for hmm0 in hmms0:
        # TODO: scaled baum
        if isScale:
            raise NotImplementedError, "Scaled baum-welch is not impl. yet"
        else:
            if algorithm == 'marginalization':
                p, iteration = \
                    hmm0.train_baumwelch_noscale(seqs, rtol, max_iter, avails)
            if algorithm == 'gluing':
                p, iteration = \
                    hmm0.train_baumwelch_gluing(seqs, rtol, max_iter, avails)
        if (p_max < p and np.isfinite(p)):
            hmm_best = copy.deepcopy(hmm0)
            p_max = p
            iter_max = iteration
        if verbose:
            print "another approximation: p=" + str(p)
            print "iteration = " + str(iteration)
            print hmm0._pi
            print hmm0._a
            print hmm0._b
    return hmm_best, iter_max

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

def estimate_hmm_params_by_seq_and_states(N, M, seqs, state_seqs):
    """ to check that sequences agrees with hmm produced it
    n -- number of hidden states
    m -- number of symbols in alphabet
    seq -- generated sequence
    states -- hidden states appeared during generation
    """
    K = len(seqs)
    pi = np.zeros(N)
    a = np.zeros(shape=(N,N))
    b = np.zeros(shape=(N,M))
    for k in range(K):
        pi_, a_, b_ = \
            _estimate_hmm_params_by_seq_and_states(N,M,seqs[k],state_seqs[k])
        pi += pi_
        a += a_
        b += b_
    return pi/K, a/K, b/K
    
def _estimate_hmm_params_by_seq_and_states(N, M, seq, state_seq):
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
    pi[state_seq[0]] = 1.0
    for t in range(T-1):
        a[state_seq[t], state_seq[t+1]] += 1.0
    a = (a.T / np.sum(a, axis=1)).T
    for t in range(T):
        b[state_seq[t], seq[t]] += 1.0
    b = (b.T / np.sum(b, axis=1)).T
    return pi, a, b
    
def classify_seqs(seqs, hmms, avails=None, isScale=False):
    if isScale:
        raise NotImplementedError, "Scaled classify_seqs is not impl. yet"
    predictions = []
    for k in range(len(seqs)):
        seq = seqs[k]
        p_max = np.finfo(np.float64).min
        s_max = 0
        for s in range(len(hmms)):
            hmm = hmms[s]
            if avails is not None:
                p = hmm.calc_likelihood_noscale([seq], [avails[k]])
            else:
                p = hmm.calc_likelihood_noscale([seq])
            if p > p_max:
                p_max = p
                s_max = s
        predictions.append(s_max)
    return predictions
    