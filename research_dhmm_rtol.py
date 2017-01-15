# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import DiscreteHMM as dhmm
import HMM

print("Starting convergence experiment...")

start_time = time.time()

#
# research params
#
dA = 0.4
n_of_launches = 1
K = 100
T = 100
T_for_dist = 500
K_for_dist = 100
hmms0_size = 5
max_iter = 10000
is_using_true_hmm_for_hmms0 = False
sig_val = 0.1
rtol_range = np.array([1e-1**i for i in range(1, 7)])
is_gaps_places_different = True
use_predefined_hmms0 = False
n_of_gaps = 0  # int(T * 0.1)
algorithm = 'marginalization'

#
# true HMM parameters
#
# hmm
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.1+dA, 0.7-dA, 0.2],
              [0.2, 0.2+dA, 0.6-dA],
              [0.8-dA, 0.1, 0.1+dA]])
b = np.array([[0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.1]])
N, M = b.shape
hmm = dhmm.DHMM(N, M, pi=pi, a=a, b=b)

#
# Optional initial approximation
#
# hmm 0
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.6, 0.2, 0.2],
              [0.2, 0.6, 0.2],
              [0.2, 0.2, 0.6]])
b = np.array([[0.2, 0.2, 0.6],
              [0.2, 0.6, 0.2],
              [0.6, 0.2, 0.2]])
hmm0 = dhmm.DHMM(N, M, pi=pi, a=a, b=b)


#
# init variables to store research results
#
if use_predefined_hmms0:
    hmms0 = [hmm0]
else:
    # one hmm is default, with equal probabilities
    hmms0 = [dhmm.DHMM(N, M, seed=i) for i in range(hmms0_size-1)]
    hmms0.append(dhmm.DHMM(N, M))


# generate gaps positions
def gen_gaps(seqs):
    np.random.seed(1)
    to_dissapears = []
    if is_gaps_places_different:
        # gaps are in different places in sequences
        for k in range(K):
            to_dissapear = np.arange(T)
            np.random.shuffle(to_dissapear)
            to_dissapears.append(to_dissapear)
    else:
        # gaps are at the same place in sequences
        to_dissapear = np.arange(T)
        np.random.shuffle(to_dissapear)
        for k in range(K):
            to_dissapears.append(to_dissapear)
    # mark some elements as availiable and others as missing
    avails = [np.full(seqs[k].shape[0], True, dtype=np.bool) for k in range(K)]
    for k in range(K):
        avails[k][to_dissapears[k][:n_of_gaps]] = False
        seqs[k][to_dissapears[k][:n_of_gaps]] = -2000
    return seqs, avails


#
# experiment
#
iters = np.zeros_like(rtol_range, dtype=np.int)
ps = np.zeros_like(rtol_range)
pi_norms = np.zeros_like(rtol_range)
a_norms = np.zeros_like(rtol_range)
b_norms = np.zeros_like(rtol_range)
dists = np.zeros_like(rtol_range)
sum_p_true = 0.0
for n_of_launch in range(n_of_launches):
    print("n_of_launch = {}".format(n_of_launch))
    print()
    seqs, state_seqs = hmm.generate_sequences(K, T, seed=n_of_launch)

    p_true = hmm.calc_loglikelihood(seqs)
    sum_p_true += p_true
    seqs_original = copy.deepcopy(seqs)
    seqs, avails = gen_gaps(seqs)

    step = 0
    for rtol in rtol_range:
        # train hmm
        np.random.seed(n_of_launch)
        hmm_trained, iter_best = \
            dhmm.train_best_hmm_baumwelch(seqs, hmms0_size, N, M,
                                          hmms0=copy.deepcopy(hmms0),
                                          rtol=rtol, max_iter=max_iter,
                                          avails=avails, algorithm=algorithm)
        p = hmm_trained.calc_loglikelihood(seqs_original)
        diff_pi = np.linalg.norm(hmm_trained._pi - hmm._pi)
        diff_a = np.linalg.norm(hmm_trained._a - hmm._a)
        diff_b = np.linalg.norm(hmm_trained._b - hmm._b)
        np.random.seed(n_of_launch)
        dist = HMM.calc_symmetric_distance(hmm_trained, hmm,
                                           T_for_dist, K_for_dist)
        ps[step] += p
        pi_norms[step] += diff_pi
        a_norms[step] += diff_a
        b_norms[step] += diff_b
        dists[step] += dist
        iters[step] += iter_best
        print("rtol = {}".format(rtol))
        print("p = {}".format(p))
        print("p_true = {}".format(p_true))
        print("diff_pi = {}".format(diff_pi))
        print("diff_a = {}".format(diff_a))
        print("diff_b = {}".format(diff_b))
        print("dist = {}".format(dist))
        # print(str(hmm_trained))
        print("iter_best = {}".format(iter_best))
        print(("--- {:.2f} minutes ---".format((time.time()-start_time) / 60)))
        print()
        step += 1

sum_p_true = sum_p_true / n_of_launches
iters = iters / n_of_launches
ps = ps / n_of_launches
pi_norms = pi_norms / n_of_launches
a_norms = a_norms / n_of_launches
b_norms = b_norms / n_of_launches
dists = dists / n_of_launches

xs = np.arange(rtol_range.size)


# draw plots
# plot all this
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font', **font)
mpl.rc('font', size=12)
plt.figure(figsize=(1920/96, 1000/96), dpi=96)

labels = ["{}, 1e-{}".format(iters[i], xs[i]+1)
          for i in range(len(rtol_range))]
print(labels)

suptitle = u"Исследование сходимости алгоритма Баума-Велша. "\
           u"Число скрытых состояний N = {}, число смесей M = {}, "\
           u"число обучающих последовательностей K = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число начальных приближений = {}, "\
           u"число запусков эксперимента n_of_launches = {} \n"\
           u"число пропусков={}, алгоритм={}" \
           .format(N, M, K, T, hmms0_size, n_of_launches, n_of_gaps, algorithm)
plt.suptitle(suptitle)
xlabel = u"Число итераций"
ax1 = plt.subplot(321)
plt.ylabel(u"Логарифм правдоподобия")
plt.xlabel(xlabel)
plt.plot(xs, ps, '-', label=u'Обученная модель')
plt.plot(xs, np.full(len(xs), sum_p_true), '--', label=u'Истинная модель')
plt.xticks(xs, labels)
plt.gca().xaxis.grid(True)
plt.legend(loc='lower right')

ax2 = plt.subplot(322)
plt.ylabel(r"$||\Pi - \Pi^*||$")
plt.xlabel(xlabel)
plt.plot(xs, pi_norms, '-')
plt.xticks(xs, labels)
plt.gca().xaxis.grid(True)

ax3 = plt.subplot(323, sharex=ax1)
plt.ylabel(r"$||A - A^*||$")
plt.xlabel(xlabel)
plt.plot(xs, a_norms, '-')
plt.xticks(xs, labels)
plt.gca().xaxis.grid(True)

ax4 = plt.subplot(324, sharex=ax2)
plt.ylabel(r"$||B - B^*||$")
plt.xlabel(xlabel)
plt.plot(xs, b_norms, '-')
plt.xticks(xs, iters)
plt.gca().xaxis.grid(True)

ax5 = plt.subplot(325, sharex=ax1)
plt.ylabel(r"$D_s(\lambda, \lambda^*)$")
plt.xlabel(xlabel)
plt.plot(xs, dists, '-')
plt.xticks(xs, labels)
plt.gca().xaxis.grid(True)

# plt.tight_layout(pad=0.0,h_pad=0.01)
plt.show()

filename = r"out\\research_dhmm_convergence_dA={}_N={}_M={}_K={}_T={}"\
            "_hmms0_size={}_launches={}_nofgaps={}_alg={}"\
           .format(dA, N, M, K, T, hmms0_size, n_of_launches, n_of_gaps,
                   algorithm)
plt.savefig(filename + ".png")
