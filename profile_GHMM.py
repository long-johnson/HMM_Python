# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import time
import numpy as np
import GaussianHMM as ghmm

K = 100
T = 100
max_iter = 10
hmms0_size = 1
hmms0 = None
is_using_true_hmm_for_hmms0 = False
sig_val = 0.1
rtol = 1e-20

#
# true HMM parameters
#
# multidimensional with mixtures
pi = np.array([0.3, 0.4, 0.3])
a = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.2, 0.6],
              [0.8, 0.1, 0.1]])
tau = np.array([[0.3, 0.4, 0.3],
                [0.3, 0.4, 0.3],
                [0.3, 0.4, 0.3]])
mu = np.array([
              [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
              [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
              ])
Z = (mu[0,0]).size
N, M = tau.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

seqs, state_seqs = hmm.generate_sequences(K, T, seed=1)

start_time = time.time()

hmm_trained, p_max, iter_best, n_of_best = \
            ghmm.train_best_hmm_baumwelch(seqs, hmms0_size, N, M, Z, hmms0=hmms0,
                                          rtol=rtol, max_iter=max_iter)
                                          
print (str(hmm_trained))

print("--- %.1f sec ---" % ((time.time()-start_time)))