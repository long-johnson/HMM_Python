# -*- coding: utf-8 -*-

"""
Various functions that apply both to Disrete HMM amd Gaussian HMM
"""

import numpy as np
import DiscreteHMM as dhmm
import GaussianHMM as ghmm


def calc_symmetric_distance(hmm1, hmm2, T, K=1, seed=None):
    """ Calculate symmetric distance between two hmms based on the logprobs.
    refer to formula (89) from "Rabiner, 1989 - A tutorial on hidden Markov..."

    Parameters
    ----------
    T : int
        length of sequences to evaluate on
    K : int
        number of sequences to evaluate on
    """
    seqs1, _ = hmm1.generate_sequences(K, T, seed=seed)
    seqs2, _ = hmm2.generate_sequences(K, T, seed=seed)
    return np.abs(_calc_distance(hmm1, hmm2, seqs2) +
                  _calc_distance(hmm2, hmm1, seqs1)) / 2


def _calc_distance(hmm1, hmm2, seqs2):
    """ Calculate assymetric distance between hmm1 and hmm2
    refer to formula (88) from "Rabiner, 1989 - A tutorial on hidden Markov..."
    NOTE: lists seqs1 and seqs2 should be of the same size and length of each
    sequence from seqs1 should be equal to the length of the corresponding seq
    from seqs2. Otherwise, the distance measure will be incorrect.
    """
    p12 = hmm1.calc_loglikelihood(seqs2)
    p22 = hmm2.calc_loglikelihood(seqs2)
    # calc total number of elements in all sequences
    # TODO: consider the case when number of elements vary from seq to seq
    n_elements = len(seqs2) * len(seqs2[0])
    return (p22 - p12) / n_elements
