# -*- coding: utf-8 -*-

import numpy as np

class MarkovChain:

    """Discrete-time markov chain (DTMC)"""    
    
    def __init__(self, N, Pi, A):
        """Constructor of an DTMC with the given parameters.
        N -- number of states
        Pi -- Initial state distribution vector
        A -- transtition probabilities matrix
        """
        self.N = N
        """Number of states in Markov chain"""
        self.Pi = Pi
        """Initial state distribution vector"""
        self.A = A
        """State transition matrix"""  
    
    @classmethod
    def default_markov_chain(cls, N):
        """Factory method of an DTMC, where all transitions are equally
        probable.
        N -- number of states
        """
        Pi = np.full(N, 1.0/N)
        A = np.full((N, N), 1.0/N)
        return cls(N, Pi, A)