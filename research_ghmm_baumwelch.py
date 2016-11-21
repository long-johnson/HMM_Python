# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import GaussianHMM as ghmm

print "Starting experiment..."

start_time = time.time()

# research params
n_of_launches = 5
rtol = 1e-20
max_iter = 300
K = 1
hmms0_size = 5
T_max = 1500
t_range = range(100,200,50) + range(200,T_max+1,100)
is_using_true_hmm_for_hmms0 = False
sig_val = 0.1

# set up true model
"""
# two-dimensional, no mixtures
pi = np.array([0.3, 0.4, 0.3])
a = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.2, 0.6],
              [0.8, 0.1, 0.1]])
tau = np.array([[1.0],
                [1.0],
                [1.0]])
mu = np.array([
              [[0.0, 0.0]],
              [[1.0, 1.0]],
              [[2.0, 2.0]],
              ])
Z = (mu[0,0]).size
N, M = tau.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)
"""
"""
# one-dimensional, mixtures
pi = np.array([0.3, 0.4, 0.3])
a = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.2, 0.6],
              [0.8, 0.1, 0.1]])
tau = np.array([[0.3, 0.4, 0.3],
                [0.3, 0.4, 0.3],
                [0.3, 0.4, 0.3]])
mu = np.array([
              [[0.0], [1.0], [2.0]],
              [[3.0], [4.0], [5.0]],
              [[6.0], [7.0], [8.0]],
              ])
Z = (mu[0,0]).size
N, M = tau.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)
"""

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


"""
# true model
pi = np.array([0.3, 0.4, 0.3])
a = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.2, 0.6],
              [0.8, 0.1, 0.1]])
tau = np.array([[1.0],
                [1.0],
                [1.0]])
mu = np.array([
              [[0.0]],
              [[1.0]],
              [[2.0]],
              ])
Z = (mu[0,0]).size
N, M = tau.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)
"""

if is_using_true_hmm_for_hmms0:
    hmm0 = copy.deepcopy(hmm)
    hmms0 = [hmm0]
else:
    hmms0 = None

xs = t_range
ps = np.full(len(t_range), 0.0)
pi_norms = np.full(len(t_range), 0.0)
a_norms = np.full(len(t_range), 0.0)
tau_norms = np.full(len(t_range), 0.0)
mu_norms = np.full(len(t_range), 0.0)
sig_norms = np.full(len(t_range), 0.0)
sum_p_true = 0.0

for n_of_launch in range(n_of_launches):
    print "n_of_launch = {}".format(n_of_launch)
    print
    seqs_full, state_seqs = hmm.generate_sequences(K, T_max, seed=n_of_launch)

    # calc for each t!
    p_true = hmm.calc_likelihood(seqs_full)
    sum_p_true += p_true
    
    step = 0
    for t in t_range:
        # truncate train sequence from full sequence
        seqs = []
        for k in range(K):
            seqs.append(seqs_full[k][:t])
        # train hmm
        np.random.seed(n_of_launch)
        hmm_trained, p_max, iter_best, n_of_best = \
            ghmm.train_best_hmm_baumwelch(seqs, hmms0_size, N, M, Z, hmms0=hmms0,
                                          rtol=rtol, max_iter=max_iter)
        p = hmm_trained.calc_likelihood(seqs_full)
        diff_pi = np.linalg.norm(hmm_trained._pi - hmm._pi)           
        diff_a = np.linalg.norm(hmm_trained._a - hmm._a)
        diff_tau = np.linalg.norm(hmm_trained._tau - hmm._tau)
        diff_mu = np.linalg.norm(hmm_trained._mu - hmm._mu)
        diff_sig = np.linalg.norm(hmm_trained._sig - hmm._sig)
        ps[step] += p
        pi_norms[step] += diff_pi
        a_norms[step] += diff_a
        tau_norms[step] += diff_tau
        mu_norms[step] += diff_mu
        sig_norms[step] += diff_sig
        print "t = {}".format(t)
        print "p = {}".format(p)
        print "p_max = {}".format(p_max)
        print "p_true = {}".format(p_true)
        print "diff_pi = {}".format(diff_pi)
        print "diff_a = {}".format(diff_a)
        print "diff_tau = {}".format(diff_tau)
        print "diff_mu = {}".format(diff_mu)
        print "diff_sig = {}".format(diff_sig)
        print "pi\n{}".format(hmm_trained._pi)
        print "a\n{}".format(hmm_trained._a)
        print "tau\n{}".format(hmm_trained._tau)
        print "mu\n{}".format(hmm_trained._mu)
        print "sig\n{}".format(hmm_trained._sig)
        print "iter_best = {}".format(iter_best)
        print "n_of_best = {}".format(n_of_best)
        print("--- %.1f minutes ---" % ((time.time()-start_time) / 60))
        print
        step += 1

sum_p_true /= n_of_launches
ps /= n_of_launches
pi_norms /= n_of_launches
a_norms /= n_of_launches
tau_norms /= n_of_launches
mu_norms /= n_of_launches
sig_norms /= n_of_launches

if np.linalg.norm(tau_norms) < 1e-2:
    tau_norms[:] = 0.0
    
# draw plots
# plot all this
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font',**font)
mpl.rc('font', size=12)
plt.figure(figsize=(1920/96, 1000/96), dpi=96)

suptitle = u"Исследование сходимости алгоритма Баума-Велша. "\
           u"Число скрытых состояний N = {}, число смесей M = {}, "\
           u"Размерность наблюдений Z = {}, \n"\
           u"число обучающих последовательностей K = {}, "\
           u"максимальная длина обучающих последовательностей Tmax = {}, \n"\
           u"число начальных приближений = {}, "\
           u"относительная невязка для останова алгоритма rtol = {}, "\
           u"максимальное число итераций iter = {}, " \
           u"число запусков эксперимента n_of_launches = {} \n"\
           u"в качестве начального приближения взята истинная модель = {}"\
           .format(N, M, Z, K, T_max, hmms0_size, rtol, max_iter, n_of_launches,
                    is_using_true_hmm_for_hmms0)
plt.suptitle(suptitle)
ax1 = plt.subplot(321)
plt.ylabel(u"Логарифм правдоподобия")
plt.xlabel(u"Длина обучающей последовательности")
plt.plot(xs, ps, '-', label=u'Обученная модель')
plt.plot(xs, np.full(len(xs), sum_p_true), ':', label=u'Истинная модель')
plt.legend(loc='lower right')

ax2 = plt.subplot(322)
plt.ylabel(r"$||\Pi - \Pi^*||$")
plt.xlabel(u"Длина обучающей последовательности")
plt.plot(xs, pi_norms, '-')

ax3 = plt.subplot(323, sharex=ax1)
plt.ylabel(r"$||A - A^*||$")
plt.xlabel(u"Длина обучающей последовательности")
plt.plot(xs, a_norms, '-')

ax4 = plt.subplot(324, sharex=ax2)
plt.ylabel(r"$||\tau - \tau^*||$")
plt.xlabel(u"Длина обучающей последовательности")
plt.plot(xs, tau_norms, '-')

ax5 = plt.subplot(325, sharex=ax1)
plt.ylabel(r"$||\mu - \mu^*||$")
plt.xlabel(u"Длина обучающей последовательности")
plt.plot(xs, mu_norms, '-')

ax6 = plt.subplot(326, sharex=ax2)
plt.ylabel(r"$||\Sigma - \Sigma^*||$")
plt.xlabel(u"Длина обучающей последовательности")
plt.plot(xs, sig_norms, '-')

#plt.tight_layout(pad=0.0,h_pad=0.01)
plt.show()

filename = "research0_gauss_N={}_M={}_Z={}_K={}_Tmax={}_hmms0_size={}_rtol={}_iter={}_launches={}"\
           "truehmm0={}"\
            .format(N, M, Z, K, T_max, hmms0_size, rtol, max_iter, n_of_launches, 
                    is_using_true_hmm_for_hmms0)
plt.savefig(filename+".png")