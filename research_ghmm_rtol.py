# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import HMM
import GaussianHMM as ghmm

print ("Starting convergence experiment...")

start_time = time.time()

#
# rtol=1e-3 is ideal for dA=0.0, sigval=0.1
# rtol=1e-4 is ideal for dA=0.1, sigval=0.1
# rtol=1e-5 is ideal for dA=0.2, sigval=0.1
#
# rtol=1e-5 (или больше) for dA=0.1, sigval=0.2
# rtol=1e-6 (или больше) for dA=0.1, sigval=0.4

#
# research params
#

n_of_launches = 1
K = 100
T = 100
T_for_dist = 500
K_for_dist = 100
hmms0_size = 5
max_iter = 10000
is_using_true_hmm_for_hmms0 = False
rtol_range = np.array([1e-1**i for i in range(1, 8)])
is_gaps_places_different = True
n_of_gaps = int(T * 0)
algorithm = 'marginalization'
#algorithm = 'viterbi'
dA = 0.0
dtau = 0.1
dmu = 0.1
dsig = 0.1
sig_val = 0.1

#
# true HMM parameters
#
pi = np.array([0.3, 0.4, 0.3])
a = np.array([[0.1+dA, 0.7-dA, 0.2],
              [0.2, 0.2+dA, 0.6-dA],
              [0.8-dA, 0.1, 0.1+dA]])
tau = np.array([[0.3+dtau, 0.4-dtau, 0.3],
                [0.3, 0.4+dtau, 0.3-dtau],
                [0.3-dtau, 0.4, 0.3+dtau]])
mu = np.array([
              [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
              [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
              ])
mu[:, :, 0] -= dmu
mu[:, :, 1] += dmu
Z = (mu[0,0]).size
N, M = tau.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * (sig_val + dsig)
hmm = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

# is_using_true_hmm_for_hmms0
if is_using_true_hmm_for_hmms0:
    hmm0 = copy.deepcopy(hmm)
    hmms0 = [hmm0]
else:
    hmms0 = None
    
    
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
        seqs[k][to_dissapears[k][:n_of_gaps]] = np.nan
    return seqs, avails


#
# experiment
#
iters = np.zeros_like(rtol_range, dtype=np.int)
ps = np.zeros_like(rtol_range)
pi_norms = np.zeros_like(rtol_range)
a_norms = np.zeros_like(rtol_range)
tau_norms = np.zeros_like(rtol_range)
mu_norms = np.zeros_like(rtol_range)
sig_norms = np.zeros_like(rtol_range)
dists = np.zeros_like(rtol_range)

sum_p_true = 0.0
for n_of_launch in range(n_of_launches):
    print ("n_of_launch = {}".format(n_of_launch))
    print
    seqs, state_seqs = hmm.generate_sequences(K, T, seed=n_of_launch)

    p_true = hmm.calc_loglikelihood(seqs)
    sum_p_true += p_true
    
    seqs_original = copy.deepcopy(seqs)
    
    seqs, avails = gen_gaps(seqs)
    
    step = 0
    for rtol in rtol_range:
        # train hmm
        np.random.seed(n_of_launch)
        hmm_trained, p_max, iter_best, n_of_best = \
            ghmm.train_best_hmm_baumwelch(seqs, hmms0_size, N, M, Z, hmms0=hmms0,
                                          rtol=rtol, max_iter=max_iter, avails=avails,
                                          algorithm=algorithm)
        p = hmm_trained.calc_loglikelihood(seqs_original)
        diff_pi = np.linalg.norm(hmm_trained._pi - hmm._pi)           
        diff_a = np.linalg.norm(hmm_trained._a - hmm._a)
        diff_tau = np.linalg.norm(hmm_trained._tau - hmm._tau)
        diff_mu = np.linalg.norm(hmm_trained._mu - hmm._mu)
        diff_sig = np.linalg.norm(hmm_trained._sig - hmm._sig)
        np.random.seed(n_of_launch)
        dist = HMM.calc_symmetric_distance(hmm_trained, hmm,
                                           T_for_dist, K_for_dist)
        ps[step] += p
        pi_norms[step] += diff_pi
        a_norms[step] += diff_a
        tau_norms[step] += diff_tau
        mu_norms[step] += diff_mu
        sig_norms[step] += diff_sig
        iters[step] += iter_best
        dists[step] += dist
        print("rtol = {}".format(rtol))
        print("p = {}".format(p))
        print("p_max = {}".format(p_max))
        print("p_true = {}".format(p_true))
        print("diff_pi = {}".format(diff_pi))
        print("diff_a = {}".format(diff_a))
        print("diff_tau = {}".format(diff_tau))
        print("diff_mu = {}".format(diff_mu))
        print("diff_sig = {}".format(diff_sig))
        print("dist = {}".format(dist))
        # print (str(hmm_trained))
        print("iter_best = {}".format(iter_best))
        print("n_of_best = {}".format(n_of_best))
        print(("--- {:.1f} minutes ---".format((time.time()-start_time) / 60)))
        
        print
        step += 1

sum_p_true /= n_of_launches
iters //= n_of_launches
ps /= n_of_launches
pi_norms /= n_of_launches
a_norms /= n_of_launches
tau_norms /= n_of_launches
mu_norms /= n_of_launches
sig_norms /= n_of_launches
dists = dists / n_of_launches

xs = np.arange(rtol_range.size)


# draw plots
# plot all this
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font',**font)
mpl.rc('font', size=12)
plt.figure(figsize=(1920/96, 1000/96), dpi=96)

labels = ["{}, 1e-{}".format(iters[i], xs[i]+1) for i in range(len(rtol_range))]
print (labels)

suptitle = u"Исследование сходимости алгоритма Баума-Велша. "\
           u"Число скрытых состояний N = {}, число смесей M = {}, "\
           u"Размерность наблюдений Z = {}, \n"\
           u"число обучающих последовательностей K = {}, "\
           u"максимальная длина обучающих последовательностей Tmax = {}, \n"\
           u"число начальных приближений = {}, "\
           u"число запусков эксперимента n_of_launches = {}, dA={}, sig_val={}\n"\
           u"в качестве начального приближения взята истинная модель = {}, "\
           u"число пропусков={}, алгоритм={}" \
           .format(N, M, Z, K, T, hmms0_size, n_of_launches, dA, sig_val,
                   is_using_true_hmm_for_hmms0, n_of_gaps, algorithm)
xlabel = u"Число итераций"

plt.suptitle(suptitle)

ax1 = plt.subplot(121)
plt.ylabel(u"Логарифм правдоподобия")
plt.xlabel(xlabel)
plt.plot(xs, ps, '-', label=u'Обученная модель')
plt.plot(xs, np.full(len(xs), sum_p_true), '--', label=u'Истинная модель')
plt.xticks(xs, labels)
plt.gca().xaxis.grid(True)
plt.legend(loc='lower right')

#ax2 = plt.subplot(322)
#plt.ylabel(r"$||\Pi - \Pi^*||$")
#plt.xlabel(xlabel)
#plt.plot(xs, pi_norms, '-')
#plt.xticks(xs, labels)
#plt.gca().xaxis.grid(True)
#
#ax3 = plt.subplot(323, sharex=ax1)
#plt.ylabel(r"$||A - A^*||$")
#plt.xlabel(xlabel)
#plt.plot(xs, a_norms, '-')
#plt.xticks(xs, labels)
#plt.gca().xaxis.grid(True)
#
#ax4 = plt.subplot(324, sharex=ax2)
#plt.ylabel(r"$||\tau - \tau^*||$")
#plt.xlabel(xlabel)
#plt.plot(xs, tau_norms, '-')
#plt.xticks(xs, iters)
#plt.gca().xaxis.grid(True)
#
#ax5 = plt.subplot(325, sharex=ax1)
#plt.ylabel(r"$||\mu - \mu^*||$")
#plt.xlabel(xlabel)
#plt.plot(xs, mu_norms, '-')
#plt.xticks(xs, labels)
#plt.gca().xaxis.grid(True)
#
#ax6 = plt.subplot(326, sharex=ax2)
#plt.ylabel(r"$||\Sigma - \Sigma^*||$")
#plt.xlabel(xlabel)
#plt.plot(xs, sig_norms, '-')
#plt.xticks(xs, labels)
#plt.gca().xaxis.grid(True)

ax5 = plt.subplot(122, sharex=ax1)
plt.ylabel(r"$D_s(\lambda, \lambda^*)$")
plt.xlabel(xlabel)
plt.plot(xs, dists, '-')
plt.xticks(xs, labels)
plt.gca().xaxis.grid(True)

plt.show()

filename = "out/ghmm_research_convergence_N={}_M={}_Z={}_K={}_T={}_hmms0_size={}_launches={}_"\
           "truehmm0={}_nofgaps={}_alg={}"\
            .format(N, M, Z, K, T, hmms0_size, n_of_launches, 
                    is_using_true_hmm_for_hmms0, n_of_gaps, algorithm)
plt.savefig(filename+".png")