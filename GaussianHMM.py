# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats
from sklearn import svm
from sklearn import preprocessing
import sklearn.mixture
import copy
from itertools import product
import StandardImputationMethods as imp


# global parameter
is_cov_diagonal = False

class GHMM:
    """Implementation of Hidden Markov Model where observation density is
    represented by a mixture of normal distributions  
    Observations are vectors of real numbers
       
       Attributes
        ----------
        N : integer
            number of hidden states
        M : integer 
            number of distribution mixture components
        Z : integer 
            dimension of observations
        pi : integer 1darray (N)
            initial state distribution vector
        a : 2darray (NxN)
            transition probabilities matrix 
        tau : 2darray (NxM)
            weights of mixture distributions 
        mu : 3darray (NxMxZ)
            means of normal distributions
        sig : 4darray (NxMxZ)
            covariation matrix of normal distributions
    """
    
    def __init__(self, n, m, z, mu, sig, pi=None, a=None, tau=None, seed=None):
        """ 
        
        Parameters
        ----------
        n : integer
            number of hidden states
        m : integer 
            number of distribution mixture components
        z : integer 
            dimension of observations
        pi : integer 1darray (N)
            initial state distribution vector
        a : 2darray (NxN)
            transition probabilities matrix 
        tau : 2darray (NxM)
            weights of mixture distributions 
        mu : 3darray (NxMxZ)
            means of normal distributions
        sig : 4darray (NxMxZ)
            covariation matrix of normal distributions
        """
        if seed is not None:
            np.random.seed(seed)  
        self._n = n
        self._m = m
        self._z = z
        # states that were not eliminated from hmm
        self._avail_states = np.full(n, True, dtype=np.bool)
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
            self._tau = np.array(tau)
        elif seed is None:
            self._tau = np.full((n,m), 1.0/m)
        else:
            self._tau = np.empty(shape=(n, m))
            for i in range(n):
                self._tau[i,:] = _generate_discrete_distribution(m)
        # TODO: add random generation of mu and sig
        self._mu = np.array(mu)
        self._sig = np.array(sig)
        
    def __str__(self):
        return "pi\n{}\n".format(self._pi) + \
               "a\n{}\n".format(self._a) + \
               "tau\n{}\n".format(self._tau) + \
               "mu\n{}\n".format(self._mu) + \
               "sig\n{}\n".format(self._sig)
        
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
        if seed is not None:
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
    
    def calc_loglikelihood(self, seqs, avails=None):
        """
        Calc likelihood of the sequences being generated by the current HMM
        
        Parameters
        ----------
        seqs : list of 2darrays (T x Z)
            observations sequences
        avails : list of boolean 1darrays (T), optional
            indicate whether element of sequence is not missing,
            i.e. True - not missing, False - is missing
            
        Returns
        -------
        likelihood : float64
            loglikelihood of the sequences being generated by the current HMM
        """
        likelihood = 0.0
        for k in range(len(seqs)):
            seq = seqs[k]
            avail = avails[k] if avails is not None \
                              else np.full(len(seq), True, dtype=np.bool)
            b, _ = self._calc_b(seq, avail)
            likelihood += self._calc_forward_scaled(b)[0]
        return likelihood
   
    def _calc_b(self, seq, avail):
        """
        Calc conditional densities of each sequence element given each HMM state
        
        Parameters
        ----------
        seq : 2darray (T x Z)
            observations sequence 
        avail : boolean 1darray (T)
            indicate whether element of sequence is not missing,
            i.e. True - not missing, False - is missing
        
        Returns
        -------
        b : 2darray (T x N)
            conditional densities for each sequence element and HMM state, i.e.
            probabilities of generating element given that hmm was in the specific state
        g : 3darray (T x N x M)
            pdf (Gaussian distribution) values for each sequence element, given
            specific hidden state and specific mixture element
        """
        N = self._n
        M = self._m
        T = seq.shape[0]
        mu = self._mu
        sig = self._sig
        tau = self._tau
        g = np.empty((T, N, M))
        global is_cov_diagonal
        for t in range(T):
            if avail[t]:
                for i in range(N):
                    for m in range(M):
                        if (tau[i,m] != 0.0):
                            temp = _my_multivariate_normal_pdf(seq[t], mu[i,m], sig[i,m], 
                                                               cov_is_diagonal = is_cov_diagonal)
                            if temp == 0.0:
                                temp = 1.0e-200 # to prevent underflow
                            g[t, i, m] = temp
                        else:
                            g[t, i, m] = 0.0
            else:
                g[t,:,:] = 1.0
        b = np.sum(tau * g, axis=2)
        return b, g
    
    def _calc_forward_scaled(self, b):
        """
        Calc scaled forward variables
        
        Parameters
        ----------
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
        T = b.shape[0]       
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
    
    def _calc_backward_scaled(self, b, c):
        """
        Calc scaled backward variables
        
        Parameters
        ----------
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
        T = b.shape[0]       
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
        
    def _calc_xi_gamma_scaled(self, b, alpha, beta):
        """ Calc xi(t,i,j), t=1..T, i,j=1..N - array of probabilities of
        being in state i and go to state j in time t given the model and seq
        Calc gamma(t,i), t=1..T, i=1..N -- array of probabilities of
        being in state i at the time t given the model and sequence
        
        Parameters
        ----------
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
        T = b.shape[0]
        N = self._n
        xi = np.empty(shape=(T-1, N, N))
        a_tr = np.transpose(self._a)
        for t in range(T-1):           
            xi[t,:,:] = (alpha[t,:] * a_tr).T * b[t+1,:] * beta[t+1,:]
        gamma = np.sum(xi, axis=2)
        return xi, gamma
     
    def _calc_gamma_m_scaled(self, b, g, gamma):
        """ Calc gamma_m(t,i,m), t=1..T, i=1..N, m=1..M -- array of probs
        of being in state i at time t and selecting m-th mixture component
        
        Parameters
        ----------
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
        T = b.shape[0]       
        tau = self._tau
        gamma_m = np.empty(shape=(T-1, N, M))
        gamma_m = g[:-1,:,:] * tau * gamma[:,:, np.newaxis] / b[:-1,:,np.newaxis]
        # nullify probs for eliminated states
        gamma_m[:-1, np.logical_not(self._avail_states), :] = 0.0
        return gamma_m
    
    def train_baumwelch(self, seqs, rtol, max_iter, avails=None):
        """ Adjust the parameters of the HMM using Baum-Welch algorithm
        
        Parameters
        ----------
        seqs : list of float64 2darrays (TxZ)
            training sequences
            Note: len(seqs) = K
        rtol : float64
            relative tolerance (stopping criterion)
        max_iter : float64, optional
            maximum number of Baum-Welch iterations (stopping criterion)
        avails : list of boolean 1darrays (T), optional
            arrays that indicate whether each element of each sequence is 
            not missing (availiable), i.e. True - not missing, False - is missing
            Note: len(avails) = K
            
        Returns
        -------
        likelihood : float64
            total likelihood of training seqs being produced by the trained HMM
        iteration : int
            iteration reached during baum-welch training process
        """
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
            a_down = np.zeros(N)
            tau_down = np.zeros(N)
            mu_up = np.zeros((N, M, Z))
            mu_sig_down = np.zeros((N, M))
            gamma_ms = np.zeros((K,T-1,N,M))
            sig_up = np.zeros((N, M, Z, Z))
            for k in range(K):   
                # expectation
                seq = seqs[k]
                avail = avails[k] if avails is not None \
                                  else np.full(seq.shape[0], True, dtype=np.bool)
                avail_t = avail[:-1]
                b, g = self._calc_b(seq, avail)
                p, alpha, c = self._calc_forward_scaled(b)
                beta = self._calc_backward_scaled(b, c)
                xi, gamma = self._calc_xi_gamma_scaled(b, alpha, beta)
                gamma_ms[k] = self._calc_gamma_m_scaled(b, g, gamma)
                # accumulating for maximization
                temp_avail_last = avail[-1]  # not to touch the last element
                avail[-1] = False            # not to touch the last element
                pi_up += gamma[0,:]
                a_up += np.sum(xi, axis=0)
                a_down += np.sum(gamma, axis=0)                
                tau_up += np.sum(gamma_ms[k][avail_t], axis=0)
                tau_down += np.sum(gamma[avail_t], axis=0)  
                mu_up += np.einsum('tnm,tz->nmz', gamma_ms[k][avail_t], seq[avail])
                mu_sig_down += np.sum(gamma_ms[k][avail_t], axis=0)
                avail[-1] = temp_avail_last  # to restore original value
            # re-estimation
            self._pi = pi_up / K
            self._a = (a_up.T / a_down).T
            self._tau = (tau_up.T / tau_down).T
            self._mu = mu_up / mu_sig_down[:,:,np.newaxis]
            # accumulating sig
            # TODO: is it possible to optimize this ...?
            for k in range(K):
                seq = seqs[k]
                avail = avails[k] if avails is not None \
                                  else np.full(seq.shape[0], True, dtype=np.bool)
                T = seq.shape[0]
                for t in range(T-1):
                    if avail[t]:
                        diff = self._mu - seq[t]
                        for i in range(N):
                            for m in range(M):
                                sig_up[i,m] += gamma_ms[k,t,i,m] * \
                                               diff[i,m] * (diff[i,m]).reshape((Z,1))
                                           #instead of np.outer()
            # sig re-estimation
            self._sig = sig_up / mu_sig_down[:,:,np.newaxis,np.newaxis]
            #print "iter = {}\nhmm params = {}\n".format(iteration, str(self))
            # remove mixture components with singular covariance matrix and
            # states where all mixtures were removed, then readjust probabilities
         
            self.normalize()
            iteration += 1
        likelihood = self.calc_loglikelihood(seqs, avails)
        return likelihood, iteration
        
    def normalize(self):
        """ remove mixture components with singular covariance matrix and
        states where all mixtures were removed, then readjust probabilities
        """
        N = self._n
        M = self._m
        pi = self._pi
        a = self._a
        tau = self._tau
        mu = self._mu
        sig = self._sig
        global is_cov_diagonal
        
        pi[np.isnan(self._pi)] = 0.0
        a[np.isnan(self._a)] = 0.0
        tau[np.isnan(self._tau)] = 0.0
        mu[np.isnan(self._mu)] = 0.0
        sig[np.isnan(self._sig)] = 0.0
        
        if is_cov_diagonal: 
            for i, m in product(range(N), range(M)):
                # make covariances diagonal again!
                sig[i,m] = np.diag(np.diag(sig[i,m]))
        for i, m in product(range(N), range(M)):
            if self._avail_states[i] and tau[i,m] != 0.0 and _is_singular(sig[i,m]):
                # save the weight and then redistribute it
                temp = tau[i,m]
                tau[i,m] = 0.0
                idx_nonzeros = np.nonzero(tau[i,:])[0]
                count_nonzeros = idx_nonzeros.size
                if count_nonzeros >= 1:
                    # redistribute weights
                    tau[i,idx_nonzeros] += temp / idx_nonzeros.size
                else:
                    # throw out state with deleted mixtures
                    self._avail_states[i] = False
                    temp = pi[i]
                    pi[i] = 0.0
                    idx_nonzeros = np.nonzero(pi)[0]
                    pi[idx_nonzeros] += temp / idx_nonzeros.size
                    # nullify inbound transitions and redistribute probs
                    for j in range(N):
                        temp = a[j,i]
                        a[j,i] = 0.0
                        idx_nonzeros = np.nonzero(pi)[0]
                        a[j,idx_nonzeros] += temp / idx_nonzeros.size
                    # nullify outbound transitions
                    a[i,:] = 0.0
        
    def decode_viterbi(self, seqs, avails=None):
        """ Infer the sequence of hidden states that were reached during generation
            multiple sequences version
        
        Parameters
        ----------
        seqs : list of float64 2darray
            observation sequence
        avail : boolean 1darray (T)
            indicate whether element of sequence is not missing,
            i.e. True - not missing, False - is missing
        
        Returns
        -------
        states_list : int 1darray
            Inferred sequence of hidden states    
        """
        K = len(seqs)
        if avails == None:
            avails = [np.full(len(seqs[k]), True, dtype=np.bool) for k in range(K)]
        states_list = []
        for k in range(K): 
            states_list.append(self._decode_viterbi(seqs[k], avails[k]))
        return states_list
        
    def _decode_viterbi(self, seq, avail):
        """ Infer the sequence of hidden states that were reached during generation
            single sequence version
        
        seq : float64 2darray
            observation sequence
        avail : boolean 1darray (T)
            indicate whether element of sequence is not missing,
            i.e. True - not missing, False - is missing
        
        Returns
        -------
        states : int 1darray
            Inferred sequence of hidden states                
        """
        T = seq.shape[0]
        N = self._n
        pi = self._pi
        log_a_tr = (np.log(self._a)).T
        b, _ = self._calc_b(seq, avail)
        log_b = np.log(b)
        psi = np.empty(shape=(T, N), dtype=np.int32)
        row_idx = np.arange(N) # to select max columns
        # initialization
        delta = np.log(pi) + log_b[0, :]     
        # recursion
        for t in range(1,T):
            temp = delta + log_a_tr
            argmax = np.argmax(temp, axis=1)
            psi[t,:] = argmax
            delta = temp[row_idx, argmax] + log_b[t, :]
        # backtracking
        q = np.empty(T, dtype=np.int32)
        q[-1] = np.argmax(delta)
        for t in reversed(range(T-1)):
            q[t] = psi[t+1, q[t+1]]
        return q
    
    def impute_by_states(self, seqs_, avails, states_list):
        """ Impute gaps according to the most probable hidden states path.
        Gap is imputed with mean that corresponds to the inferred hidden state
        and the most probable mixture component
        
        Parameters
        ----------
        seqs_ : list of float64 2darray
            observation sequences with gaps
        avails : list of boolean 1darrays (T)
            arrays that indicate whether element of sequence is not missing,
            i.e. True - not missing, False - is missing
        states_list : int 1darray
            Inferred sequence of hidden states
            
        Returns
        -------
        seqs : list of float64 2darray
            imputed observation sequences
        """
        seqs = copy.deepcopy(seqs_)
        K = len(seqs)
        mu = self._mu
        sig = self._sig
        tau = self._tau
        global is_cov_diagonal
        for k in range(K):
            seq = seqs[k]
            avail = avails[k]
            states = states_list[k]
            for t in np.where(avail==False)[0]:
                # impute gaps by drawing from a mixture of multinormal distributions
                state = states[t]
                mix_component = _get_sample_discrete_distr(tau[state,:])
                seq[t] = np.random.multivariate_normal(mu[state, mix_component],
                                                        sig[state, mix_component])
        return seqs
    
    def train_bauwelch_impute_viterbi(self, seqs, rtol, max_iter, avails, 
                                      isRegressive=False):
        """ Train HMM with Baum-Welch by imputing missing observations using 
        Viterbi decoder.
        
        Parameters
        ----------
        seqs : list of float64 2darrays (TxZ)
            training sequences
            Note: len(seqs) = K
        rtol : float64
            relative tolerance (stopping criterion)
        max_iter : float64
            maximum number of Baum-Welch iterations (stopping criterion)
        avails : list of boolean 1darrays (T)
            arrays that indicate whether each element of each sequence is 
            not missing, i.e. True - not missing, False - is missing
        isRegressive : bool, optional
            true: imputation begins from the start of sequence after each imputed gap
            false: imputation performed once
        
        Returns
        -------
        p : float64
            total likelihood of training seqs being produced by the trained HMM
        it : int
            iteration reached during baum-welch training process
        """
        # Choosing the imputation mode:
        if isRegressive:
            raise NotImplementedError("Regressive is not implemented yet")
        else:
            hmm0 = copy.deepcopy(self)
            hmm0.train_baumwelch(seqs, rtol, max_iter, avails)
            # TODO: check correcteness
            states_decoded = hmm0.decode_viterbi(seqs, avails)
            # TODO: check correcteness
            seqs_imputed = hmm0.impute_by_states(seqs, avails, states_decoded)
            p, it = self.train_baumwelch(seqs_imputed, rtol, max_iter)
        return p, it
        
    def train_bauwelch_impute_mean(self, seqs, rtol, max_iter, avails, params=[10]):
        """ Train HMM with Baum-Welch by restoring gaps using mean imputation
        
        Parameters
        ----------
        seqs : list of float64 2darrays (TxZ)
            training sequences
            Note: len(seqs) = K
        rtol : float64
            relative tolerance (stopping criterion)
        max_iter : float64
            maximum number of Baum-Welch iterations (stopping criterion)
        avails : list of boolean 1darrays (T)
            arrays that indicate whether each element of each sequence is 
            not missing, i.e. True - not missing, False - is missing
        params : list of one item
            number of neighbours (default: 10)
            
        Returns
        -------
        p : float64
            total likelihood of training seqs being produced by the trained HMM
        it : int
            iteration reached during baum-welch training process
        """
        n_neighbours = params[0]
        seqs_imp, avails_imp = imp.impute_by_n_neighbours(seqs, avails, n_neighbours,
                                              is_middle=True, method="mean")
        # in case some gaps were not imputed
        seqs_imp = imp.impute_by_whole_seq(seqs_imp, avails_imp, method="mean")
        p, it = self.train_baumwelch(seqs_imp, rtol, max_iter)
        return p, it
        
    def train_baumwelch_gluing(self, seqs, rtol, max_iter, avails):
        """ Glue segments between gaps together and then train Baum-Welch
        
        Parameters
        ----------
        seqs : list of float64 2darrays (TxZ)
            training sequences
            Note: len(seqs) = K
        rtol : float64
            relative tolerance (stopping criterion)
        max_iter : float64
            maximum number of Baum-Welch iterations (stopping criterion)
        avails : list of boolean 1darrays (T)
            arrays that indicate whether each element of each sequence is availiable
        
        Returns
        -------
        p : float64
            total likelihood of training seqs being produced by the trained HMM
        it : int
            iteration reached during baum-welch training process
        """
        K = len(seqs)
        # remove all gaps and just glue remaining segments together
        seqs_glued = []
        for k in range(K):
            glued = seqs[k][avails[k]]  # fancy indexing
            seqs_glued.append(glued)
        p, it = self.train_baumwelch(seqs_glued, rtol, max_iter)
        return p, it
        
    def calc_derivatives(self, seqs, avails=None, wrt=None,
                         algorithm_gaps='marginalization',
                         n_neighbours=10):
        """ Calculate derivatives of loglikelihood function for the given sequences
        with respect to each HMM parameter
        
        Parameters
        ----------
        seqs : list (K) of float 2darrays (TxZ)
            given sequences
        avails : list (K) of boolean 1darrays (T), optional
            array that indicate the available observations in seq
        wrt : list of strings {'pi', 'a', 'tau', 'mu', 'sig'}
            with respect to which paramerers the derivatives should be taken
        Returns
        -------
        derivs : float 2darray K x (N + NxN + NxM + NxMxZ + NxMxZxZ)
            derivatives of loglikelihood function for each of the given sequences
            with respect to each HMM parameter
        """
        assert(algorithm_gaps in ['marginalization', 'viterbi', 'gluing', 'mean'])
        N, M, Z = self._mu.shape
        K = len(seqs)
        derivatives = [np.empty(0) for k in range(K)]
        for k in range(K):
            seq = seqs[k]
            avail = avails[k] if avails is not None\
                    else np.full(len(seq), True, dtype=np.bool)
            if algorithm_gaps == 'viterbi':
                states = self.decode_viterbi([seq], avails=[avail])[0]
                seq = self.impute_by_states([seq], [avail], [states])[0]
            if algorithm_gaps == 'gluing':
                seq = seq[avail]    # fancy indexing
            if algorithm_gaps == 'mean':
                seqs_imp, avails_imp = \
                    imp.impute_by_n_neighbours([seq], [avail], n_neighbours,
                                               method='mean')
                seq = imp.impute_by_whole_seq(seqs_imp, avails_imp,
                                              method='mean')[0]
            if algorithm_gaps in ['viterbi', 'gluing', 'mean']:
                avail = np.full(len(seq), True, dtype=np.bool)
            b, g = self._calc_b(seq, avail)
            _, alpha, c = self._calc_forward_scaled(b)
            if wrt is None or 'pi' in wrt:
                d_loglike_wrt_pi = self._calc_derivs_pi(seq, avail, b, c, alpha)
                derivatives[k] = np.append(derivatives[k], d_loglike_wrt_pi)
            if wrt is None or 'a' in wrt:
                d_loglike_wrt_a = self._calc_derivs_a(seq, avail, b, c, alpha)
                derivatives[k] = np.append(derivatives[k], d_loglike_wrt_a)
            if wrt is None or 'tau' in wrt:
                d_loglike_wrt_tau = self._calc_derivs_tau(seq, avail, b, c, alpha, g)
                derivatives[k] = np.append(derivatives[k], d_loglike_wrt_tau)
            if wrt is None or 'mu' in wrt:
                d_loglike_wrt_mu = self._calc_derivs_mu(seq, avail, b, c, alpha, g)
                derivatives[k] = np.append(derivatives[k], d_loglike_wrt_mu)
            if wrt is None or 'sig' in wrt:
                d_loglike_wrt_sig = self._calc_derivs_sig(seq, avail, b, c, alpha, g)
                derivatives[k] = np.append(derivatives[k], d_loglike_wrt_sig)
        return np.array(derivatives)

    def _calc_derivs_pi(self, seq, avail, b, c, alpha):
        N = self._n
        T = len(seq)
        d_loglike_wrt_pi = np.empty(N)
        d_b_wrt_pi = np.zeros((T, N))
        d_a_wrt_pi =  np.zeros((N, N))
        for i in range(N):
            d_alpha0_wrt_pi = np.zeros(N)   # alpha0 unscaled
            d_alpha0_wrt_pi[i] = b[0, i]
            d_loglike_wrt_pi[i] = self._calc_d_loglike_wrt_nu(avail, b, c, alpha,
                                                              d_alpha0_wrt_pi,
                                                              d_b_wrt_pi,
                                                              d_a_wrt_pi)
        return d_loglike_wrt_pi
    
    def _calc_derivs_a(self, seq, avail, b, c, alpha):
        N = self._n
        T = len(seq)
        d_loglike_wrt_a = np.empty((N,N))
        d_alpha0_wrt_a = np.zeros(N)   # alpha0 unscaled
        d_b_wrt_a = np.zeros((T, N))
        for i, j in product(range(N), range(N)):
            d_a_wrt_a =  np.zeros((N, N))
            d_a_wrt_a[i, j] = 1.0
            d_loglike_wrt_a[i, j] = self._calc_d_loglike_wrt_nu(avail, b, c, alpha,
                                                                d_alpha0_wrt_a,
                                                                d_b_wrt_a,
                                                                d_a_wrt_a)
        return d_loglike_wrt_a
    
    def _calc_derivs_tau(self, seq, avail, b, c, alpha, g):
        N, M, pi = self._n, self._m, self._pi
        T = len(seq)
        d_loglike_wrt_tau = np.empty((N, M))
        d_a_wrt_tau =  np.zeros((N, N))
        for i, m in product(range(N), range(M)):
            d_b_wrt_tau = np.zeros((T, N))
            d_b_wrt_tau[:, i] = g[:, i, m]
            d_alpha0_wrt_tau = np.zeros(N)   # alpha0 unscaled
            d_alpha0_wrt_tau[i] = pi[i] * d_b_wrt_tau[0, i]
            d_loglike_wrt_tau[i, m] = self._calc_d_loglike_wrt_nu(avail, b, c, alpha,
                                                                  d_alpha0_wrt_tau,
                                                                  d_b_wrt_tau,
                                                                  d_a_wrt_tau)
        return d_loglike_wrt_tau
    
    def _calc_derivs_mu(self, seq, avail, b, c, alpha, g):
        N, M, Z = self._n, self._m, self._mu.shape[2]
        pi, tau, mu, sig = self._pi, self._tau, self._mu, self._sig
        T = len(seq)
        d_loglike_wrt_mu = np.empty((N, M, Z))
        d_a_wrt_mu = np.zeros((N, N))
        for i, m in product(range(N), range(M)): # loop over derivatives index
            sig_inv = np.linalg.inv(sig[i, m])
            full_d_b_wrt_mu = np.zeros((T, N, Z))
            for t in range(T): # loop over t (explicitly) and i (subtly)
                if avail[t]:
                    full_d_b_wrt_mu[t, i] = tau[i, m] * g[t, i, m] * \
                                            (sig_inv @ (seq[t] - mu[i, m]))
                else:
                    full_d_b_wrt_mu[t, i] = 0.0
            for z in range(Z):  # loop over derivatives index
                d_b_wrt_mu = full_d_b_wrt_mu[:, :, z]
                d_alpha0_wrt_mu = np.zeros(N)   # alpha0 unscaled
                d_alpha0_wrt_mu[i] = pi[i] * d_b_wrt_mu[0, i]
                d_loglike_wrt_mu[i, m, z] = self._calc_d_loglike_wrt_nu(avail, b, c, alpha,
                                                                        d_alpha0_wrt_mu,
                                                                        d_b_wrt_mu,
                                                                        d_a_wrt_mu)
        return d_loglike_wrt_mu

    def _calc_derivs_sig(self, seq, avail, b, c, alpha, g):
        N, M, Z = self._n, self._m, self._mu.shape[2]
        pi, tau, mu, sig = self._pi, self._tau, self._mu, self._sig
        T = len(seq)
        d_loglike_wrt_sig = np.empty((N, M, Z, Z))
        d_a_wrt_sig = np.zeros((N, N))
        for i, m in product(range(N), range(M)): # loop over derivatives index
            sig_inv = np.linalg.inv(sig[i, m])
            full_d_b_wrt_sig = np.zeros((T, N, Z, Z))
            for t in range(T): # loop over t (explicitly) and i (subtly)
                if avail[t]:
                    temp = (seq[t] - mu[i, m])[np.newaxis].T # column vector
                    full_d_b_wrt_sig[t, i] = 0.5 * tau[i, m] * g[t, i, m] * \
                                             (sig_inv @ temp @ temp.T @ sig_inv - sig_inv)
                else:
                    full_d_b_wrt_sig[t, i] = 0.0
            for z1, z2 in product(range(Z), range(Z)):  # loop over derivatives index
                d_b_wrt_sig = full_d_b_wrt_sig[:, :, z1, z2]
                d_alpha0_wrt_sig = np.zeros(N)   # alpha0 unscaled
                d_alpha0_wrt_sig[i] = pi[i] * d_b_wrt_sig[0, i]
                d_loglike_wrt_sig[i, m, z1, z2] = \
                    self._calc_d_loglike_wrt_nu(avail, b, c, alpha, d_alpha0_wrt_sig,
                                                d_b_wrt_sig, d_a_wrt_sig)
        return d_loglike_wrt_sig

    def _calc_d_loglike_wrt_nu(self, avail, b, c, alpha,
                               d_alpha0_wrt_nu, d_b_wrt_nu, d_a_wrt_nu):
        """ Calculate derivative of loglikelihood function with respect to
        some parameter nu
    
        Parameters
        ----------
        b : float 1darray (TxN)
            _
        c : float 1darray (T)
            _
        alpha : float 2darray (TxN)
            _
        avail : bool 1darray (T)
            _
        d_alpha0_wrt_nu : float 1darray (N)
            _
        d_b_wrt_nu : float 1darray (TxN)
            _
        d_a_wrt_nu : float 1darray (NxN)
            _
        """
        #print("---!!!---")
        N = self._n
        a_T = np.transpose(self._a)
        d_a_wrt_nu_T = np.transpose(d_a_wrt_nu)
        T = len(b)
        d_alphatilde_wrt_nu = np.empty((T, N))   # alpha temp (with tilde)
        d_alphatilde_wrt_nu[0] = d_alpha0_wrt_nu
        d_c_wrt_nu = np.empty(T)
        for t in range(1, T):
            d_c_wrt_nu[t-1] = -(c[t-1] ** 2) * np.sum(d_alphatilde_wrt_nu[t-1])
            alphatilde = alpha[t-1] / c[t-1]
            d_alpha_wrt_nu = d_c_wrt_nu[t-1] * alphatilde + d_alphatilde_wrt_nu[t-1] * c[t-1]
            d_alphatilde_wrt_nu[t] = np.sum(d_alpha_wrt_nu * a_T +
                                     alpha[t-1] * d_a_wrt_nu_T, axis=1) * b[t] + \
                                     np.sum(alpha[t-1] * a_T, axis=1) * d_b_wrt_nu[t]
        d_c_wrt_nu[-1] = -(c[-1] ** 2) * np.sum(d_alphatilde_wrt_nu[-1])
        return -np.sum(d_c_wrt_nu / c)


def train_best_hmm_baumwelch(seqs, hmms0_size, N, M, Z, avails=None, rtol=1e-1, max_iter=None,
                             algorithm='marginalization', hmms0=None, verbose=False,
                             initial_guess='gmm', covariance_type='diag'):
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
    algorithm : {'marginalization', 'gluing', 'viterbi', 'mean'}
        which algorithm should be used to handle missing observations
    avails : list of boolean 1darrays (T)
            arrays that indicate whether each element of each sequence is 
            not missing (availiable), i.e. True - not missing, False - is missing
    hmms0 : list of GHMMs, optional
        list of initial approximations of HMM parameters
        !note: len(hmms0) == hmms0_size must be fulfilled!
        if not specified, hmms0_size approximations will be generated
    rtol : float64, optional
        relative tolerance (stopping criterion)
    max_iter : float64, optional
        maximum number of Baum-Welch iterations (stopping criterion)
    initial_guess : string {'gmm', 'uniform'}
        determines how to make initial guess of mu and sig params
    verbose : bool, optional
        controls whether some debug info should be printed to stdout
    
    Returns
    -------
    hmm_best : GHMM
        best trained hmm or None
    p_max : float64
        likelihood for the best hmm
    iter_best : int
        number of iterations to train the best hmm 
    n_of_best : int
        number of the initial approximation that gave the best hmm
    """
    assert algorithm in ('marginalization', 'gluing', 'viterbi', 'mean'),\
                         "Invalid algorithm '{}'".format(algorithm)
    if hmms0 is None:
        if initial_guess == 'uniform':
            mu_est, sig_est = estimate_mu_sig_uniformly(seqs, N, M, avails)
        if initial_guess == 'gmm':
            mu_est, sig_est = estimate_mu_sig_gmm(seqs, N, M, avails,
                                                  covariance_type=covariance_type)
        if verbose:
            print ("mu_est: {}".format(mu_est))
            print ("sig_est: {}".format(sig_est))
        hmms = [GHMM(N, M, Z, mu_est, sig_est, seed=np.random.randint(10000))
                for i in range(hmms0_size-1)]       
        # standard pi, a, tau parameters
        hmms.append(GHMM(N,M,Z,mu_est,sig_est))
    else:
        hmms = copy.deepcopy(hmms0)
    p_max = np.finfo(np.float64).min # minimal value possible
    hmm_best = None
    iter_best = -1 # count number of iters for the best hmm
    n_of_best = -1  # number of the best hmm
    n_of_approx = 0
    # calc and choose the best hmm estimate
    for hmm in hmms:
        if algorithm == 'marginalization':
            p, iteration = hmm.train_baumwelch(seqs, rtol, max_iter, avails)
        if algorithm == 'gluing':
            p, iteration = hmm.train_baumwelch_gluing(seqs, rtol, max_iter, avails)
        if algorithm == 'viterbi':
            p, iteration = hmm.train_bauwelch_impute_viterbi(seqs, rtol, max_iter, avails)
        if algorithm == 'mean':
            p, iteration = hmm.train_bauwelch_impute_mean(seqs, rtol, max_iter, avails)
        if (p_max < p and np.isfinite(p)):
            hmm_best = hmm
            p_max = p
            iter_best = iteration
            n_of_best = n_of_approx
        if verbose:
            print ("Baum: n of approximation = {}".format(n_of_approx))
            print ("p={}".format(p))
            print ("iteration = ".format(iteration))
            print (str(hmm))
            print
        n_of_approx += 1
    return hmm_best, p_max, iter_best, n_of_best


def _form_train_data_for_SVM(hmms, seqs_list, avails_list=None, wrt=None,
                             algorithm_gaps='marginalization', n_neighbours=10):
    """
    wrt : list of strings {‘pi’, ‘a’, ‘tau’, ‘mu’, ‘sig’}
        with respect to which paramerers the derivatives should be taken
    """
    n_of_classes = len(hmms)
    # define training samples
    X = []
    for hmm in hmms:
        # vertical block of sequence derivatives
        Xvblock = []
        # for each set of sequences
        for s in range(n_of_classes):
            seqs = seqs_list[s]
            avails = avails_list[s] if avails_list is not None else None
            Xvblock.append(hmm.calc_derivatives(seqs, avails, wrt=wrt, 
                                                algorithm_gaps=algorithm_gaps,
                                                n_neighbours=n_neighbours))
        X.append(np.concatenate(Xvblock, axis=0))
    X = np.concatenate(X, axis=1)
    # define labels
    y = []
    for s in range(n_of_classes):
        y.append(np.full(len(seqs_list[s]), fill_value=s, dtype=np.int))
    y = np.concatenate(y, axis=0)
    return X, y


def train_svm_classifier(hmms, seqs_list, clf, avails_list=None, X=None, y=None, wrt=None,
                         algorithm_gaps='marginalization', n_neighbours=10):
    """ Train svm classifier that classifies sequences based on their 
    derivatives of likelihood function with respect to hmm params
    
    Parameters
    ----------
    hmms : list of hmms representing each of the competing classes
        all hmms must have the same structure
    seqs_list : list (len(hmms)) of lists (K) of float 2darrays (TxZ)
        list of lists of sequences belonging to each class
    clf : sklearn.svm.SVC
        parameters of SVM classifier
    avails_list : list (len(hmms)) of lists (K) of bool 1darrays (T), optional
        _
    wrt : list of strings {‘pi’, ‘a’, ‘tau’, ‘mu’, ‘sig’}
        with respect to which paramerers the derivatives should be taken
        
    len(hmms) == len(seqs_list) == len(avails_list)  
    
    Returns
    -------
    clf : sklearn.svm.SVC
        SVM classifier
    scaler : scaler
        scaler
    """
    assert(len(hmms) == len(seqs_list) and
           (avails_list is None or len(seqs_list) == len(avails_list)))
    # define training samples
    if X is None or y is None:
        X, y = _form_train_data_for_SVM(hmms, seqs_list, avails_list, wrt=wrt,
                                        algorithm_gaps=algorithm_gaps,
                                        n_neighbours=n_neighbours)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    clf.fit(X, y)
    return clf, scaler


def _form_class_data_for_SVM(hmms, seqs, avails=None, wrt=None,
                             algorithm_gaps='marginalization', n_neighbours=10):
    """
    wrt : list of strings {‘pi’, ‘a’, ‘tau’, ‘mu’, ‘sig’}
        with respect to which paramerers the derivatives should be taken
    """
    X = [hmm.calc_derivatives(seqs, avails, wrt=wrt, algorithm_gaps=algorithm_gaps,
                              n_neighbours=n_neighbours) for hmm in hmms]
    X = np.concatenate(X, axis=1)
    return X


def classify_seqs_svm(seqs, hmms, clf, scaler, avails=None, wrt=None,
                      algorithm_gaps='marginalization', n_neighbours=10, X=None):
    """ Predict class label for each seq based on derivatives of likelihood
    function for that seq using SVM classifier
    
    Parameters
    ----------
    seqs : list of 2darrays (TxZ)
        sequences to be classified
    hmms : list of GHMMs 
        list of hmms each of which corresponds to a class
    clf : sklearn.svm.SVC
        SVM classifier
    wrt : list of strings {‘pi’, ‘a’, ‘tau’, ‘mu’, ‘sig’}
        with respect to which paramerers the derivatives should be taken
        
    Returns
    -------
    predictions : list of ints
        list of class labels
    """
    if X is None:
        X = _form_class_data_for_SVM(hmms, seqs, avails, wrt=wrt,
                                     algorithm_gaps=algorithm_gaps)
    X = scaler.transform(X)
    return clf.predict(X)
    

def estimate_mu_sig_uniformly(seqs, N, M, avails=None):
    """ Estimate values of mu and sig basing on the sequences.
    mu elements are uniformly scattered from min to max seq element
    sig matrixes are diagonal scaled accordingly to min and max seq elements
    
    Parameters
    ----------
    seqs : list of 2darrays (TxZ)
        list of training sequences
    N, M : int
        probable number of states and mixture components in HMM
    avails : list of boolean 1darrays (T), optional
        arrays that indicate whether each element of each sequence is 
        not missing (availiable), i.e. True - not missing, False - is missing
    
    Returns
    -------
    mu : 3darray (NxMxZ)
        means of normal distributions 
    sig : 4darray (NxMxZxZ)
        covariation matrix of normal distributions
    """
    # TODO: add more clever heuristics to this procedure
    K = len(seqs)
    Z = len(seqs[0][0])
    if avails is None:
        avails = [np.full(shape=seqs[k].shape[0], fill_value=True, dtype=np.bool) for k in range(K)]
    mu = np.empty((N*M,Z))
    sig = np.empty((N,M,Z,Z))
    min_val = np.min([np.min(seqs[k][avails[k]], axis=0) for k in range(K)], axis=0)
    max_val = np.max([np.max(seqs[k][avails[k]], axis=0) for k in range(K)], axis=0)
    step = (max_val - min_val) / (N*M)
    val = min_val + step/2.0
    for i in range(N*M):
        mu[i] = val
        val += step
    mu = np.reshape(mu, newshape=(N,M,Z))
    # TODO: scale sig matrixes
    for i in range(N):
        for j in range(M):
            sig[i,j] = np.eye(Z)
    return mu, sig


def estimate_mu_sig_gmm(seqs, N, M, avails=None, n_init=5,
                        covariance_type='diag'):
    """ Estimate some HMMs params, namely: means of gaussian mixtures (mu) and
    covariance matrixes of gaussian mixtures given the observation points.
    The estimation is done using gaussian mixture models (GMM).
    This is needed primarily for a good initial guess of HMM parameters.
    
    Parameters
    ----------
    seqs : list (K) of ndarrays (TxZ)
        observation sequences generated by the modelled process
    N, M : int
        probable number of states and mixture components in HMM
    avails : list (K) of boolean 1darrays (T), optional
        missing observations for each sequence
    n_init : int, optional
        Number GMM of initializations to perform. the best result is kept
    covariance_type : {'spherical', 'diag', 'full'}
        Sets the shape and properties of covariance matrix
    Returns
    -------
    mu : 3darray (NxMxZ)
        means of normal distributions 
    sig : 4darray (NxMxZxZ)
        covariation matrix of normal distributions
    """
    assert(covariance_type in ['spherical', 'diag', 'full'])
    points = []
    if avails is None:
        for seq in seqs:
            points += seq.tolist()
    else:
        for seq, avail in zip(seqs, avails):
            points += seq[avail].tolist()
    points = np.array(points)
    gmm = sklearn.mixture.GMM(n_components=N, covariance_type=covariance_type,
                              n_init=n_init)
    pred = gmm.fit_predict(points)
    Z = len(seqs[0][0])
    mu = np.empty((N, M, Z))
    sig = np.empty((N, M, Z, Z))
    for i in range(N):
        local_points = points[pred == i]
        gmm = sklearn.mixture.GMM(n_components=M, covariance_type=covariance_type,
                                  n_init=n_init)
        gmm.fit_predict(local_points)
        mu[i] = np.array(gmm.means_)
        if covariance_type == 'full':
            sig[i] = np.array(gmm.covars_)
        else:
            sig_tmp = np.array(gmm.covars_)
            for m in range(N):
                sig[i, m] = np.diag(sig_tmp[m])
    return mu, sig


def classify_seqs_mlc(seqs, hmms, avails=None, algorithm_gaps='marginalization',
                      n_neighbours=10):
    """ Classify sequences using maximum likelihood classifier (mlc)
    Label each seq from seqs with number of the hmm which suits seq better
        'suits' i.e. has the biggest likelihood of generating tht sequence
    
    Parameters
    ----------
    seqs : list of 2darrays (TxZ)
        sequences to be classified
    hmms : list of GHMMs 
        list of hmms each of which corresponds to a class
    avails : list of boolean 1darrays (T), optional
        arrays that indicate whether each element of each sequence is availiable
    algorithm_gaps : algorithm to be used to fight missing values
    n_neighbours : how many neighbours to account for when imputing by the mean
        of neighbours (works only with 'mean' algorithm)
        
    Returns
    -------
    predictions : int 1darray
        array of class labels
    """
    assert algorithm_gaps in ['marginalization', 'gluing', 'viterbi', 'mean',
                              'viterbi_advanced1', 'viterbi_advanced2']
    if avails is None:
        avails = [np.full(len(seqs[k]), True, dtype=np.bool) for k in range(len(seqs))]

    predictions = []
    for k in range(len(seqs)):
        seq = copy.deepcopy(seqs[k])
        avail = copy.deepcopy(avails[k])
        if algorithm_gaps[:-1] == 'viterbi_advanced':
            # viterbi imputation with advanced decision rule
            # calc probs of sequence imputed by each of the HMMs being 
            # generated by each of these HMMs
            probs = np.empty((len(hmms), len(hmms)))
            # label of hmm to impute
            for label_imp in range(len(hmms)):
                hmm_imp = hmms[label_imp]
                states = hmm_imp.decode_viterbi([seq], avails=[avail])[0]
                seq = hmm_imp.impute_by_states([seq], [avail], [states])[0]
                # label of hmm to calc probability
                for label_prob in range(len(hmms)):
                    probs[label_imp, label_prob] = hmms[label_prob].calc_loglikelihood([seq])
            # advanced decision rule
            for label_checked in range(len(hmms)):
                labels = list(range(len(hmms)))
                labels.remove(label_checked)
                is_label_checked_best = True
                # check the conditions for a confident prediction
                for label_1, label_2 in product(range(len(hmms)), labels):
                    if not probs[label_1, label_checked] > probs[label_1, label_2]:
                        is_label_checked_best = False
                        break
                if is_label_checked_best:
                    label_best = label_checked
                    break
            # check the more general condition for a less confident prediction
            if not is_label_checked_best:
                if algorithm_gaps == 'viterbi_advanced1':
                    label_best = np.argmax(np.diag(probs))
                if algorithm_gaps == 'viterbi_advanced2':
                    diag = np.diag(probs)
                    np.fill_diagonal(probs, 0.0)
                    criteria = diag - np.sum(probs, axis=1)
                    label_best = np.argmax(criteria)
        else:
            p_max = np.finfo(np.float64).min
            label_best = 0
            for label in range(len(hmms)):
                hmm = hmms[label]
                if algorithm_gaps == 'marginalization':
                    p = hmm.calc_loglikelihood([seq], [avail])
                if algorithm_gaps == 'viterbi':
                    states = hmm.decode_viterbi([seq], avails=[avail])[0]
                    seq = hmm.impute_by_states([seq], [avail], [states])[0]
                    p = hmm.calc_loglikelihood([seq])
                if algorithm_gaps == 'gluing':
                    glued = seq[avail]  # fancy indexing
                    p = hmm.calc_loglikelihood([glued])
                if algorithm_gaps == 'mean':
                    seqs_imp, avails_imp = \
                        imp.impute_by_n_neighbours([seq], [avail], n_neighbours,
                                                   method='mean')
                    seq = imp.impute_by_whole_seq(seqs_imp, avails_imp,
                                                  method='mean')[0]
                    p = hmm.calc_loglikelihood([seq])
                if p > p_max:
                    p_max = p
                    label_best = label
        predictions.append(label_best)
    return np.array(predictions, dtype=np.int)

    
def estimate_hmm_params_by_seq_and_states(mu, sig, seqs, state_seqs):
    """ to check that sequences agrees with hmm produced it
    
    Parameters
    ----------
    mu : float 3darray (NxMxZ)
        means of normal distributions
    sig : float 4darray (NxMxZ)
        covariation matrix of normal distributions
    seq : float 2darray (TxZ)
        generated sequence
    states : int 1darray (T)
        hidden states appeared during generation
    """
    N, M, Z = mu.shape
    K = len(seqs)
    pi = np.zeros(N)
    a = np.zeros(shape=(N,N))
    tau = np.zeros(shape=(N,M))
    for k in range(K):
        pi_, a_, tau_ = \
            _estimate_hmm_params_by_seq_and_states(mu, sig, seqs[k], state_seqs[k])
        pi += pi_
        a += a_
        tau += tau_
    return pi/K, a/K, tau/K

def _estimate_hmm_params_by_seq_and_states(mu, sig, seq, state_seq):
    N, M, Z = mu.shape
    T = seq.shape[0]
    pi = np.zeros(N)
    a = np.zeros(shape=(N,N))
    tau = np.zeros(shape=(N,M))
    # estimate pi
    scores = np.empty(shape=(N,M))
    for n in range(N):
        for m in range(M):
            # TODO: optimize
            scores[n, m] = \
                _my_multivariate_normal_pdf(seq[0], mu[n,m], sig[n,m])
    #state = np.argmax(np.sum(scores, axis=1))
    state = state_seq[0]
    pi[state] = 1.0
    mixture = np.argmax(scores[state])
    tau[state, mixture] += 1.0
    # estimate a
    state_prev = state
    for t in range(1,T):
        for n in range(N):
            for m in range(M):
                # TODO: optimize
                scores[n, m] = \
                    _my_multivariate_normal_pdf(seq[t], mu[n,m], sig[n,m])
        #state = np.argmax(np.sum(scores, axis=1))
        state = state_seq[t]
        a[state_prev, state] += 1.0
        mixture = np.argmax(scores[state])
        tau[state, mixture] += 1.0
        state_prev = state
    a = (a.T / np.sum(a, axis=1)).T
    tau = (tau.T / np.sum(tau, axis=1)).T
    return pi, a, tau

def _generate_discrete_distribution(n):
    """ Generate n values > 0.0, whose sum equals 1.0
    """
    xs = np.array(sp.stats.uniform.rvs(size=n))
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
    
def _my_multivariate_normal_pdf(x, mu, cov, cov_is_diagonal=False):
        """
        Calcs pdf of multivariate normal distribution.
        Doesn't allow singular covariation matrixes.
        
        Parameters
        ----------
        cov_is_diagonal : bool
            if True then cov is considered to be a diagonal matrix (more effective)
            
        Returns
        -------
        pdf : float64
            pdf of multivariate normal distribution at point x
        """
        k = x.size
        if cov_is_diagonal:
            diag = np.diag(cov)
            det = np.prod(diag)
            diag = 1.0 / diag
            cov_inv = np.diag(diag)
        else: # general case
            try:                
                cov_inv = np.linalg.inv(cov)
                det = np.linalg.det(cov)
            except np.linalg.LinAlgError:
                return 0.0
        diff = x - mu        
        part1 = 1.0 / (np.sqrt((2.0*np.pi)**k * det))
        part2 = -0.5 * np.dot(np.dot(diff, cov_inv), diff)
        return part1 * np.exp(part2)

def _is_singular(x):
    """ Effectively checks whether square matrix is singular.
    Is faster than linalg.cond(x) < 1.0 / sys.float_info.epsilon
    """
    try:
        #np.linalg.inv(x)
        sp.stats.multivariate_normal.pdf(x[0], x[0], x)
    except np.linalg.LinAlgError:
        return True
    return False