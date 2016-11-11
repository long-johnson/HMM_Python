# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import time
import GaussianHMM as ghmm
import StandardImputationMethods as stdimp

# experiment parameters
ghmm.is_cov_diagonal = False
# lcoal parameters
n_of_launches = 10
T = 100
K_train = 100
K_class = 100
rtol = 1e-5
max_iter = 100
hmms0_size = 5
sig_val = 0.1
dA = 0.2
use_predefined_hmms0 = False
is_gaps_places_different = True
is_verbose = False
is_using_true_hmm_for_hmms0 = False

#filename = "ultimate_diagcov_"+"_dA"+str(dA)+"_t"+str(T)+"_k"+str(K_train)+"_initrand"+\
#   str(hmms0_size*np.logical_not(use_predefined_hmms0))\
#    +"_rtol"+str(rtol)+"_iter"+str(max_iter)+"_x"+str(n_of_launches)
    
#gaps_range = range(0,T,T/10)
#gaps_range = range(0,100,25) + range(100,600,50) + [575] + [590]
gaps_range = range(0,100,10)
#gaps_range = [10]

# hmm 1

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
hmm1 = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

# hmm 2
pi = np.array([0.3, 0.4, 0.3])
a = np.array([[0.1+dA, 0.7-dA, 0.2],
              [0.2, 0.2+dA, 0.6-dA],
              [0.8-dA, 0.1, 0.1+dA]])
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
hmm2 = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

# hmm0 (predefined hmm)
"""
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.3, 0.6, 0.1],
              [0.3, 0.1, 0.6],
              [0.6, 0.2, 0.2]])
tau = np.array([[0.3, 0.4, 0.3],[0.3, 0.4, 0.3],[0.3, 0.4, 0.3]])

mu = np.array([
              [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
              [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
              ])
Z = (mu[0,0]).size
N, M = tau.shape
sig = np.full(shape=(N,M,Z,Z), fill_value=0.0)
diag = np.diag(np.full(Z, fill_value=sig_val))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = diag
hmm0 = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)
"""

def evaluate_training(ps, pi_norms, a_norms, tau_norms, mu_norms, sig_norms,
                      class_percent, step, seed, start_time,
                      hmm1, hmm2, seqs_train_orig1, seqs_train_orig2,
                      train_seqs1, train_seqs2, hmms0_size, N, M, Z,
                      hmms0=None, algorithm='marginalization', rtol=None,
                      max_iter=100, avails1=None, avails2=None, verbose=False):
    """ Code to evaluate hmm training performance on given sequences
    """
    np.random.seed(seed)
    hmm_trained1, _, iter1, n_of_best1 = \
        ghmm.train_best_hmm_baumwelch(train_seqs1, hmms0_size, N, M, Z, hmms0=hmms0,
                                            algorithm=algorithm, rtol=rtol,
                                            max_iter=max_iter, avails=avails1,
                                            verbose=verbose)
    np.random.seed(seed)
    hmm_trained2, _, iter2, n_of_best2 = \
        ghmm.train_best_hmm_baumwelch(train_seqs2, hmms0_size, N, M, Z, hmms0=hmms0,
                                            algorithm=algorithm, rtol=rtol,
                                            max_iter=max_iter, avails=avails2,
                                            verbose=verbose)
    if hmm_trained1 is None or hmm_trained2 is None:
        ps[step] += 0.0
        pi_norms[step] += 1.5
        a_norms[step] += 2.0
        tau_norms[step] += 2.0
        mu_norms[step] += 2.0
        sig_norms[step] += 2.0
        class_percent[step] += 50.0
        global filename
        with open(filename+"_log.txt", "a") as f:
            f.write("Bad training: n_of_launch={}, n_of_gaps={}: hmm_trained1 is None = {},"
                    "hm_trained2 is None = {}, algorithm = {}\n"\
                    .format(n_of_launch, n_of_gaps, hmm_trained1 is None, 
                            hmm_trained2 is None, algorithm))
    else:
        # diff between 1st trained model and 1st true model
        diff_pi1 = np.linalg.norm(hmm_trained1._pi-hmm1._pi)           
        diff_a1 = np.linalg.norm(hmm_trained1._a-hmm1._a)
        diff_tau1 = np.linalg.norm(hmm_trained1._tau-hmm1._tau)
        diff_mu1 = np.linalg.norm(hmm_trained1._mu-hmm1._mu)
        diff_sig1 = np.linalg.norm(hmm_trained1._sig-hmm1._sig)
        # diff between 2nd trained model and 2nd true model
        diff_pi2 = np.linalg.norm(hmm_trained2._pi-hmm2._pi)           
        diff_a2 = np.linalg.norm(hmm_trained2._a-hmm2._a)
        diff_tau2 = np.linalg.norm(hmm_trained2._tau-hmm2._tau)
        diff_mu2 = np.linalg.norm(hmm_trained2._mu-hmm2._mu)
        diff_sig2 = np.linalg.norm(hmm_trained2._sig-hmm2._sig)
        # some kind of average diff
        #diff_a = (diff_a1 + diff_a2) / 2.0
        #diff_b = (diff_b1 + diff_b2) / 2.0
        print "n_of_gaps " + str(n_of_gaps)
        print algorithm
        print "model1"
        print str(hmm_trained1)
        print "model2"
        print str(hmm_trained2)
        loglikelihood1 = hmm_trained1.calc_likelihood(seqs_train1)
        loglikelihood2 = hmm_trained2.calc_likelihood(seqs_train2)
        print "loglikelihood: {} / {}".format(loglikelihood1, loglikelihood2)
        print "loglikelihood true: {} / {}".format(loglikelihood_true1, loglikelihood_true2)
        print "norm of pi diff = " + str(diff_pi1) + " / " + str(diff_pi2)
        print "norm of A diff = " + str(diff_a1) + " / " + str(diff_a2)
        print "norm of TAU diff = " + str(diff_tau1) + " / " + str(diff_tau2)
        print "norm of MU diff = " + str(diff_mu1) + " / " + str(diff_mu2)
        print "norm of SIG diff = " + str(diff_sig1) + " / " + str(diff_sig2)
        print "Iterations: " + str(iter1) + " / " + str(iter2)
        print "n_of_best: {} / {}".format(n_of_best1, n_of_best2)
        # classification
        class_res1 = ghmm.classify_seqs(seqs_class1, [hmm_trained1, hmm_trained2])
        class_res2 = ghmm.classify_seqs(seqs_class2, [hmm_trained1, hmm_trained2])
        percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
        # update
        ps[step] += loglikelihood1
        pi_norms[step] += diff_pi1
        a_norms[step] += diff_a1
        tau_norms[step] += diff_tau1
        mu_norms[step] += diff_mu1
        sig_norms[step] += diff_sig1
        class_percent[step] += percent
        print str(percent) + " %"
    print("--- %.1f minutes ---" % ((time.time()-start_time) / 60))

#
# research
#

if ghmm.is_cov_diagonal:
    postfix = "diagcov"
else:
    postfix = "mypdf"
filename = "ultimate_{}_N={}_M={}_Z={}_Ktrain={}_T={}_hmms0_size={}_rtol={}_iter={}_launches={}_"\
           "truehmm0={}_diagcov={}"\
            .format(postfix, N, M, Z, K_train, T, hmms0_size, rtol, max_iter, n_of_launches, 
                    is_using_true_hmm_for_hmms0, ghmm.is_cov_diagonal)
    
with open(filename+"_log.txt", "w") as f:
    f.write(filename+"\n")

start_time = time.time()
if use_predefined_hmms0:
    hmms0 = []
else:
    hmms0 = None
# prepare variables to accumulate experiment data
# TODO: make a routine?
xs = gaps_range
class_percent_best = np.full(len(gaps_range), 0.0) # if classified by true models
p_true = 0.0 # loglikelihood on true model for complete sequences
# marginalization
ps_marg = np.full(len(gaps_range), 0.0)
pi_norms_marg = np.full(len(gaps_range), 0.0)
a_norms_marg  = np.full(len(gaps_range), 0.0)
tau_norms_marg  = np.full(len(gaps_range), 0.0)
mu_norms_marg = np.full(len(gaps_range), 0.0)
sig_norms_marg = np.full(len(gaps_range), 0.0)
class_percent_marg = np.full(len(gaps_range), 0.0)
# viterbi imputation
ps_viterbi = np.full(len(gaps_range), 0.0)
pi_norms_viterbi = np.full(len(gaps_range), 0.0)
a_norms_viterbi  = np.full(len(gaps_range), 0.0)
tau_norms_viterbi  = np.full(len(gaps_range), 0.0)
mu_norms_viterbi = np.full(len(gaps_range), 0.0)
sig_norms_viterbi = np.full(len(gaps_range), 0.0)
class_percent_viterbi = np.full(len(gaps_range), 0.0)
# gluing
ps_gluing= np.full(len(gaps_range), 0.0)
pi_norms_gluing = np.full(len(gaps_range), 0.0)
a_norms_gluing  = np.full(len(gaps_range), 0.0)
tau_norms_gluing  = np.full(len(gaps_range), 0.0)
mu_norms_gluing = np.full(len(gaps_range), 0.0)
sig_norms_gluing = np.full(len(gaps_range), 0.0)
class_percent_gluing = np.full(len(gaps_range), 0.0)
# mean imputation
ps_mean = np.full(len(gaps_range), 0.0)
pi_norms_mean = np.full(len(gaps_range), 0.0)
a_norms_mean  = np.full(len(gaps_range), 0.0)
tau_norms_mean  = np.full(len(gaps_range), 0.0)
mu_norms_mean = np.full(len(gaps_range), 0.0)
sig_norms_mean = np.full(len(gaps_range), 0.0)
class_percent_mean = np.full(len(gaps_range), 0.0)

# make several launches
for n_of_launch in range(n_of_launches):
    # generate new training sequences
    seqs_train1, state_seqs_train1 = hmm1.generate_sequences(K_train, T, seed=n_of_launch)
    seqs_train2, state_seqs_train2 = hmm2.generate_sequences(K_train, T, seed=n_of_launch)
    # generate new sequences to classify
    seqs_class1, _ = hmm1.generate_sequences(K_class, T, seed=n_of_launch)
    seqs_class2, _ = hmm2.generate_sequences(K_class, T, seed=n_of_launch)
    # calc best likelihood by true model on complete sequences
    loglikelihood_true1 = hmm1.calc_likelihood(seqs_train1)
    loglikelihood_true2 = hmm2.calc_likelihood(seqs_train2)
    p_true += loglikelihood_true1
    
    # prepare indices in sequence array to dissapear
    # TODO: make a routine!
    to_dissapears1 = []
    to_dissapears2 = []
    np.random.seed(n_of_launch)
    # generate gaps postitons
    if is_gaps_places_different:
        # gaps are in different places in sequences
        for k in range(K_train):
            to_dissapear = np.arange(T)
            np.random.shuffle(to_dissapear)
            to_dissapears1.append(to_dissapear)
            np.random.shuffle(to_dissapear)
            to_dissapears2.append(to_dissapear)
    else:
         # gaps are at the same place in sequences
        to_dissapear1 = np.arange(T)
        np.random.shuffle(to_dissapear1)
        to_dissapear2 = np.arange(T)
        np.random.shuffle(to_dissapear2)
        for k in range(K_train):
            to_dissapears1.append(to_dissapear1)
            to_dissapears2.append(to_dissapear2)
    
        
    # the experiment
    step = 0
    for n_of_gaps in gaps_range:
        # mark some elements as missing.
        # TODO: make a routine!
        # hmm 1
        avails1 = [np.full(seqs_train1[k].shape[0], True, dtype=np.bool) for k in range(K_train)]
        train_seqs1 = [np.array(seqs_train1[k]) for k in range(K_train)]
        for k in range(K_train):
            avails1[k][to_dissapears1[k][:n_of_gaps]] = False
            train_seqs1[k][to_dissapears1[k][:n_of_gaps]] = np.nan
        # hmm 2
        avails2 = [np.full(seqs_train2[k].shape[0], True, dtype=np.bool) for k in range(K_train)]
        train_seqs2 = [np.array(seqs_train2[k]) for k in range(K_train)]
        for k in range(K_train):
            avails2[k][to_dissapears2[k][:n_of_gaps]] = False
            train_seqs2[k][to_dissapears2[k][:n_of_gaps]] = np.nan  
            
        # best classification by true models 
        class_res1 = ghmm.classify_seqs(seqs_class1, [hmm1, hmm2])
        class_res2 = ghmm.classify_seqs(seqs_class2, [hmm1, hmm2])
        percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
        class_percent_best[step] += percent
        print "Best percent is " + str(percent) + " %"
        
        #
        # marginalization
        #        
        evaluate_training(ps_marg, pi_norms_marg, a_norms_marg, tau_norms_marg, 
                          mu_norms_marg, sig_norms_marg, class_percent_marg,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train1, seqs_train2,
                          train_seqs1, train_seqs2, hmms0_size, N, M, Z,
                          hmms0, algorithm='marginalization', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        
        #
        # viterbi
        #
        evaluate_training(ps_viterbi, pi_norms_viterbi, a_norms_viterbi, tau_norms_viterbi, 
                          mu_norms_viterbi, sig_norms_viterbi, class_percent_viterbi,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train1, seqs_train2,
                          train_seqs1, train_seqs2, hmms0_size, N, M, Z,
                          hmms0, algorithm='viterbi', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        
        #
        # gluing
        #
        evaluate_training(ps_gluing, pi_norms_gluing, a_norms_gluing, tau_norms_gluing, 
                          mu_norms_gluing, sig_norms_gluing, class_percent_gluing,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train1, seqs_train2,
                          train_seqs1, train_seqs2, hmms0_size, N, M, Z,
                          hmms0, algorithm='gluing', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        
        #
        # mean imputation
        #
        evaluate_training(ps_mean, pi_norms_mean, a_norms_mean, tau_norms_mean, 
                          mu_norms_mean, sig_norms_mean, class_percent_mean,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train1, seqs_train2,
                          train_seqs1, train_seqs2, hmms0_size, N, M, Z,
                          hmms0, algorithm='mean', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        step += 1

# get the average value
class_percent_best /= n_of_launches

ps_true = np.full(len(gaps_range), fill_value=(p_true / n_of_launches))

ps_marg /= n_of_launches
pi_norms_marg /= n_of_launches
a_norms_marg /= n_of_launches
tau_norms_marg /= n_of_launches
mu_norms_marg /= n_of_launches
sig_norms_marg /= n_of_launches
class_percent_marg /= n_of_launches

ps_viterbi /= n_of_launches
pi_norms_viterbi /= n_of_launches
a_norms_viterbi /= n_of_launches
tau_norms_viterbi /= n_of_launches
mu_norms_viterbi /= n_of_launches
sig_norms_viterbi /= n_of_launches
class_percent_viterbi /= n_of_launches

ps_gluing /= n_of_launches
pi_norms_gluing /= n_of_launches
a_norms_gluing /= n_of_launches
tau_norms_gluing /= n_of_launches
mu_norms_gluing /= n_of_launches
sig_norms_gluing /= n_of_launches
class_percent_gluing /= n_of_launches

ps_mean /= n_of_launches
pi_norms_mean /= n_of_launches
a_norms_mean /= n_of_launches
tau_norms_mean /= n_of_launches
mu_norms_mean /= n_of_launches
sig_norms_mean /= n_of_launches
class_percent_mean /= n_of_launches

# plot all this
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font',**font)
mpl.rc('font', size=12)
plt.figure(figsize=(1920/96, 1000/96), dpi=96)

suptitle = u"Исследование алгоритмов обучения СММ по посл-тям с пропусками "\
           u"Число скрытых состояний N = {}, число смесей M = {}, "\
           u"Размерность наблюдений Z = {}, \n"\
           u"число обучающих последовательностей K_train = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число начальных приближений = {}, "\
           u"относительная невязка для останова алгоритма rtol = {}, "\
           u"максимальное число итераций iter = {}, " \
           u"число запусков эксперимента n_of_launches = {} \n"\
           u"в качестве начального приближения взята истинная модель = {}"\
           .format(N, M, Z, K_train, T, hmms0_size, rtol, max_iter, n_of_launches,
                    is_using_true_hmm_for_hmms0)
plt.suptitle(suptitle)

ax1 = plt.subplot(421)
plt.ylabel(u"Логарифм правдоподобия")
plt.xlabel(u"Процент пропусков")

line1=plt.plot(xs, ps_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, ps_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, ps_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, ps_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")
#line5=plt.plot(xs, ps_true, '-', label=u"Истинная СММ без пропусков")

ax2 = plt.subplot(422)
plt.ylabel(r"$||\Pi - \Pi^*||$")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, pi_norms_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, pi_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, pi_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, pi_norms_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")

ax3 = plt.subplot(423, sharex=ax1)
plt.ylabel(r"$||A - A^*||$")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, a_norms_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, a_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, a_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, a_norms_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")

ax4 = plt.subplot(424, sharex=ax2)
plt.ylabel(r"$||\tau - \tau^*||$")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, tau_norms_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, tau_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, tau_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, tau_norms_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")

ax5 = plt.subplot(425, sharex=ax1)
plt.ylabel(r"$||\mu - \mu^*||$")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, mu_norms_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, mu_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, mu_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, mu_norms_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")

ax6 = plt.subplot(426, sharex=ax2)
plt.ylabel(r"$||\Sigma - \Sigma^*||$")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, sig_norms_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, sig_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, sig_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, sig_norms_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")

ax7 = plt.subplot(427, sharex=ax1)
plt.ylabel(u"Верно распознанные, %")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, class_percent_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, class_percent_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, class_percent_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, class_percent_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")
line5 = plt.plot(xs, class_percent_best, '-', dash_capstyle='round',  lw=2.0, label=u"Истинные модели")

plt.figlegend((line1[0], line2[0], line3[0], line4[0], line5[0]), 
              (u"Маргинализация",u"Склеивание",u"Витерби", u"Среднее", u"Истинные модели"),
              loc = 'center right')
#plt.tight_layout(pad=0.0,h_pad=0.01)
plt.show()

plt.savefig(filename+".png")

to_file = np.asarray([xs,ps_true,ps_marg, ps_gluing, ps_viterbi, ps_mean,
                      pi_norms_marg, pi_norms_gluing, pi_norms_viterbi, pi_norms_mean,
                      a_norms_marg, a_norms_gluing, a_norms_viterbi, a_norms_mean,
                      tau_norms_marg, tau_norms_gluing, tau_norms_viterbi, tau_norms_mean,
                      mu_norms_marg, mu_norms_gluing, mu_norms_viterbi, mu_norms_mean,
                      sig_norms_marg, sig_norms_gluing, sig_norms_viterbi, sig_norms_mean,
                      class_percent_marg, class_percent_gluing, class_percent_viterbi, class_percent_mean])
np.savetxt(filename+".csv", to_file.T, delimiter=';', 
           header="xs;ps_true;ps_marg;ps_gluing;ps_viterbi;ps_mean;"
                  "pi_norms_marg;pi_norms_gluing;pi_norms_viterbi;pi_norms_mean;"
                  "a_norms_marg;a_norms_gluing;a_norms_viterbi;a_norms_mean;"
                  "tau_norms_marg;tau_norms_gluing;tau_norms_viterbi;tau_norms_mean;"
                  "mu_norms_marg;mu_norms_gluing;mu_norms_viterbi;mu_norms_mean;"
                  "sig_norms_marg;sig_norms_gluing;sig_norms_viterbi;sig_norms_mean;"
                  "class_percent_marg;class_percent_gluing;class_percent_viterbi;class_percent_mean")

print("--- %.1f minutes ---" % ((time.time()-start_time) / 60))

#xs,ps1,ps_glue1,ps_viterbi1, a_norms1,a_norms_glue1,a_norms_viterbi1,b_norms1,\
#b_norms_glue1,b_norms_viterbi1, class_percent, class_percent_glue, class_percent_viterbi\
# = np.loadtxt(filename+".csv", delimiter=';', unpack=True)