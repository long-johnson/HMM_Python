# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import copy
import matplotlib.pyplot as plt
import DiscreteHMM as dhmm

print "Starting experiment..."

# research params
seed_gen = 457
seed_init_approx = 228
rtol = None
max_iter = 50
K = 1
T_max = 1000
t_range = range(50,500,50) + range(500,T_max+1,100)
# set up true model
pi = np.array([1.0, 0.0])
a = np.array([[0.7, 0.3],
              [0.3, 0.7]])
b = np.array([[0.8, 0.2],
              [0.1, 0.9]])
n, m = b.shape
hmm = dhmm.DHMM(n, m, pi=pi, a=a, b=b)
# initial approximation
hmm0 = dhmm.DHMM(n, m, seed=seed_init_approx)

seqs_full, state_seqs = hmm.generate_sequences(K, T_max, seed=seed_gen)

# check that sequences correspond to model
pi_est, a_est, b_est = \
    dhmm.estimate_hmm_params_by_seq_and_states(n, m, seqs_full, state_seqs)

xs = []
ps = []
pi_norms = []
a_norms = []
b_norms = []
for t in t_range:
    # truncate train sequence from full sequence
    seqs = []
    for k in range(K):
        seqs.append(seqs_full[k][:t])
    # train hmm
    hmm_trained = copy.deepcopy(hmm0)
    _, iteration = \
        hmm_trained.train_baumwelch_noscale(seqs, rtol=rtol, max_iter=max_iter)
    p = hmm_trained.calc_likelihood_noscale(seqs_full)
    diff_pi = np.linalg.norm(hmm_trained._pi-hmm._pi)
    diff_a = np.linalg.norm(hmm_trained._a-hmm._a)
    diff_b = np.linalg.norm(hmm_trained._b-hmm._b)
    xs.append(t)
    ps.append(p)
    #ps.append(hmm_trained.calc_likelihood_noscale(seqs))
    pi_norms.append(diff_pi)
    a_norms.append(diff_a)
    b_norms.append(diff_b)
    print hmm_trained._a
    print hmm_trained._b
    print "t = " + str(t)
    print "iter = " + str(iteration)
    print "p=" + str(p)
    
# draw plots
plt.subplot(311)
plt.ylabel("ln(p)")
plt.xlabel("T")
plt.plot(xs, np.log(ps))
#plt.subplot(222)
#plt.ylabel("||Pi*-Pi||")
#plt.xlabel("T, Length of training sequence")
#plt.plot(xs, pi_norms)
plt.subplot(312)
plt.ylabel("||A*-A||")
plt.xlabel("T")
plt.plot(xs,a_norms)
#plt.plot(xs,a_qual)
plt.subplot(313)
plt.ylabel("||B*-B||")
plt.xlabel("T")
plt.plot(xs,b_norms)
#plt.plot(xs,b_qual)
plt.show()
plt.savefig('research_0_baum.png')