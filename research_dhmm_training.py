# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import copy
import DiscreteHMM as dhmm
import HMM

start_time = time.time()

dA = 0.3
rtol = 1e-3
max_iter = 1000
T = 100
K = 100
T_for_dist = 500
K_for_dist = 100
K_class = 100
hmms0_size = 5
n_of_launches = 5
use_predefined_hmms0 = False
is_gaps_places_different = True
is_verbose = False
gaps_range = list(range(0, 100, 10))
out_dir = "out/"
filename = out_dir + "dhmm_ultimate_dA{}_T{}_K{}_initrand{}_rtol{}_iter{}_x{}"\
           .format(dA, T, K, hmms0_size*np.logical_not(use_predefined_hmms0),
                   rtol, max_iter, n_of_launches)
#
# True HMMs
#
# hmm 1
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.2, 0.6],
              [0.8, 0.1, 0.1]])
b = np.array([[0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.1]])
N, M = b.shape
hmm1 = dhmm.DHMM(N, M, pi=pi, a=a, b=b)

# hmm 2
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.1+dA, 0.7-dA, 0.2],
              [0.2, 0.2+dA, 0.6-dA],
              [0.8-dA, 0.1, 0.1+dA]])
b = np.array([[0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.1]])
hmm2 = dhmm.DHMM(N, M, pi=pi, a=a, b=b)

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


def evaluate_classification(step, hmm1, hmm2,
                            seqs1, seqs2, avails1=None, avails2=None,
                            algorithm='marginalization',  n_neighbours=10,
                            verbose=False):
    """ Code to evaluate hmm training performance on given sequences
    n_neighbours works only with ‘mode’ algorithm
    
    Returns
    -------
    percent of correctly classified sequences
    """
    print("algorithm = {}".format(algorithm))
    class_res1 = dhmm.classify_seqs(seqs1, [hmm1, hmm2], avails=avails1,
                                    algorithm=algorithm)
    class_res2 = dhmm.classify_seqs(seqs2, [hmm1, hmm2], avails=avails2,
                                    algorithm=algorithm)
    return 100.0 * (class_res1.count(0) + class_res2.count(1)) / (2.0 * K)

    
def evaluate_training(ps, dists, pi_norms, a_norms, b_norms, class_percent,
                      class_gaps_percent,
                      step, seed, start_time,
                      hmm1, hmm2, seqs_train_orig1, seqs_train_orig2,
                      train_seqs1, train_seqs2,
                      seqs_class_gaps1, seqs_class_gaps2, avails_gaps1, avails_gaps2,
                      hmms0_size, N, M,
                      hmms0=None, algorithm='marginalization', rtol=None,
                      max_iter=100, avails1=None, avails2=None, verbose=False):
    """ Code to evaluate hmm training performance on given sequences
    """
    np.random.seed(seed)
    hmm_trained1, iter1 = \
        dhmm.train_best_hmm_baumwelch(train_seqs1, hmms0_size, N, M,
                                      hmms0=hmms0, algorithm=algorithm,
                                      rtol=rtol, max_iter=max_iter,
                                      avails=avails1, verbose=verbose)
    np.random.seed(seed)
    hmm_trained2, iter2 = \
        dhmm.train_best_hmm_baumwelch(train_seqs2, hmms0_size, N, M,
                                      hmms0=hmms0, algorithm=algorithm,
                                      rtol=rtol, max_iter=max_iter,
                                      avails=avails2, verbose=verbose)
    # if training error occured
    if hmm_trained1 is None or hmm_trained2 is None:
        ps[step] += ps[step-1] if step >= 1 else -10000.0
        
        pi_norms[step] += 1.5
        a_norms[step] += 2.0
        b_norms[step] += 2.0
        class_percent[step] += 50.0
        global filename
        with open(filename+"_log.txt", "a") as f:
            f.write("Bad training: n_of_launch={}, n_of_gaps={}:"
                    "hmm_trained1 is None = {},"
                    "hm_trained2 is None = {}, algorithm = {}\n"
                    .format(n_of_launch, n_of_gaps, hmm_trained1 is None,
                            hmm_trained2 is None, algorithm))
        return
    # diff between 1st trained model and 1st true model
    diff_pi1 = np.linalg.norm(hmm_trained1._pi-hmm1._pi)
    diff_a1 = np.linalg.norm(hmm_trained1._a-hmm1._a)
    diff_b1 = np.linalg.norm(hmm_trained1._b-hmm1._b)
    # diff between 2nd trained model and 2nd true model
    diff_pi2 = np.linalg.norm(hmm_trained2._pi-hmm2._pi)
    diff_a2 = np.linalg.norm(hmm_trained2._a-hmm2._a)
    diff_b2 = np.linalg.norm(hmm_trained2._b-hmm2._b)
    loglikelihood1 = hmm_trained1.calc_loglikelihood(seqs_train_orig1)
    loglikelihood2 = hmm_trained2.calc_loglikelihood(seqs_train_orig2)
    # classification
    class_res1 = dhmm.classify_seqs(seqs_class1, [hmm_trained1, hmm_trained2])
    class_res2 = dhmm.classify_seqs(seqs_class2, [hmm_trained1, hmm_trained2])
    class_percent_tmp = 100.0 * (class_res1.count(0)+class_res2.count(1)) / (2.0*K_class)
    # loglikelihood distance
    dist1 = HMM.calc_symmetric_distance(hmm1, hmm_trained1,
                                        T_for_dist, K_for_dist)
    dist2 = HMM.calc_symmetric_distance(hmm2, hmm_trained2,
                                        T_for_dist, K_for_dist)
    # classification with gaps
    class_gaps_percent_tmp = evaluate_classification(step, hmm_trained1, hmm_trained2,
                                                     seqs_class_gaps1,
                                                     seqs_class_gaps2,
                                                     avails_gaps1, avails_gaps2,
                                                     algorithm)
    # update
    dists[step] += dist1
    ps[step] += loglikelihood1
    pi_norms[step] += diff_pi1
    a_norms[step] += diff_a1
    b_norms[step] += diff_b1
    class_percent[step] += class_percent_tmp
    class_gaps_percent[step] += class_gaps_percent_tmp

    # print
    print("n_of_gaps {}".format(n_of_gaps))
    print(algorithm)
    print("loglikelihood: {} / {}".format(loglikelihood1, loglikelihood2))
    print("norm of pi diff = {} / {}".format(diff_pi1, diff_pi2))
    print("norm of A diff = {} / {}".format(diff_a1, diff_a2))
    print("norm of B diff = {} / {}".format(diff_b1, diff_b2))
    print("distances: {} / {}".format(dist1, dist2))
    print("Iterations: {} / {}".format(iter1, iter2))
    print("Correctly classified {} %".format(class_percent_tmp))
    print("class_gaps_percent = {} %".format(class_gaps_percent_tmp))
    print("--- {:.1f} minutes ---".format((time.time()-start_time) / 60))


def gen_gaps_positions(K, T, is_gaps_places_different):
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
    return to_dissapears


def make_missing_values(seqs_train_orig, to_dissapears):
    avails = [np.full_like(seqs_train_orig[i], True, dtype=np.bool)
              for i in range(K)]
    seqs_train = copy.deepcopy(seqs_train_orig)
    for k in range(K):
        avails[k][to_dissapears[k][:n_of_gaps]] = False
        seqs_train[k][to_dissapears[k][:n_of_gaps]] = -20000
    return seqs_train, avails

#
# set specific initial hmm or generate initial hmms randomly
#
if use_predefined_hmms0:
    hmms0 = [hmm0]
else:
    # one hmm is default, with equal probabilities
    hmms0 = [dhmm.DHMM(N, M, seed=i) for i in range(hmms0_size-1)]
    hmms0.append(dhmm.DHMM(N, M))

#
# init variables to store research results
#
xs = np.array(gaps_range, dtype=np.int)

class_percent_best = 0.0
p_true = 0.0

ps_marg = np.zeros_like(xs, dtype=np.float)
pi_norms_marg = np.zeros_like(ps_marg)
a_norms_marg = np.zeros_like(ps_marg)
b_norms_marg = np.zeros_like(ps_marg)
class_percent_marg = np.zeros_like(ps_marg)
dists_marg = np.zeros_like(ps_marg)
class_gaps_percent_marg = np.zeros_like(ps_marg)

ps_viterbi = np.zeros_like(ps_marg)
pi_norms_viterbi = np.zeros_like(ps_marg)
a_norms_viterbi = np.zeros_like(ps_marg)
b_norms_viterbi = np.zeros_like(ps_marg)
class_percent_viterbi = np.zeros_like(ps_marg)
dists_viterbi = np.zeros_like(ps_marg)
class_gaps_percent_viterbi = np.zeros_like(ps_marg)

ps_gluing = np.zeros_like(ps_marg)
pi_norms_gluing = np.zeros_like(ps_marg)
a_norms_gluing = np.zeros_like(ps_marg)
b_norms_gluing = np.zeros_like(ps_marg)
class_percent_gluing = np.zeros_like(ps_marg)
dists_gluing = np.zeros_like(ps_marg)
class_gaps_percent_gluing = np.zeros_like(ps_marg)

ps_mode = np.zeros_like(ps_marg)
pi_norms_mode = np.zeros_like(ps_marg)
a_norms_mode = np.zeros_like(ps_marg)
b_norms_mode = np.zeros_like(ps_marg)
class_percent_mode = np.zeros_like(ps_marg)
dists_mode = np.zeros_like(ps_marg)
class_gaps_percent_mode = np.zeros_like(ps_marg)

#
# Research
#
for n_of_launch in range(n_of_launches):
    # generate new training sequences
    seqs_train_orig1, _ = hmm1.generate_sequences(K, T, seed=n_of_launch)
    seqs_train_orig2, _ = hmm2.generate_sequences(K, T, seed=n_of_launch)
    
    # generate gaps postitons
    np.random.seed(n_of_launch)
    to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
    to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)
    to_dissapears_class1 = gen_gaps_positions(K_class, T, is_gaps_places_different)
    to_dissapears_class2 = gen_gaps_positions(K_class, T, is_gaps_places_different)

    # generate sequences to classify
    seqs_class1, _ = hmm1.generate_sequences(K_class, T, seed=n_of_launch+1)
    seqs_class2, _ = hmm2.generate_sequences(K_class, T, seed=n_of_launch+1)

    # calc best likelihood by true model on complete sequences
    loglikelihood_true1 = hmm1.calc_loglikelihood(seqs_train_orig1)
    loglikelihood_true2 = hmm2.calc_loglikelihood(seqs_train_orig2)
    p_true += loglikelihood_true1

    # best classification by true models
    class_res1 = dhmm.classify_seqs(seqs_class1, [hmm1, hmm2])
    class_res2 = dhmm.classify_seqs(seqs_class2, [hmm1, hmm2])
    percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
    class_percent_best += percent
    print("Best percent is {} %".format(percent))
    print()

    # the experiment
    step = 0
    for n_of_gaps in gaps_range:
        # mark some elements of seqs as missing
        # hmm 1
        seqs_train1, avails1 = make_missing_values(seqs_train_orig1,
                                                   to_dissapears1)
        # hmm 2
        seqs_train2, avails2 = make_missing_values(seqs_train_orig2,
                                                   to_dissapears2)
        # mark some elements of seqs to be classified as missing
        # hmm 1
        seqs_class_gaps1, avails_gaps1 = make_missing_values(seqs_class1,
                                                             to_dissapears1)
        # hmm 2
        seqs_class_gaps2, avails_gaps2 = make_missing_values(seqs_class2,
                                                             to_dissapears2)
        #
        # marginalization
        #
        evaluate_training(ps_marg, dists_marg, pi_norms_marg, a_norms_marg,
                          b_norms_marg, class_percent_marg, class_gaps_percent_marg,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train_orig1, seqs_train_orig2,
                          seqs_train1, seqs_train2,
                          seqs_class_gaps1, seqs_class_gaps2,
                          avails_gaps1, avails_gaps2,
                          hmms0_size, N, M,
                          hmms0, algorithm='marginalization', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        #
        # viterbi
        #
        evaluate_training(ps_viterbi, dists_viterbi, pi_norms_viterbi,
                          a_norms_viterbi, b_norms_viterbi,
                          class_percent_viterbi, class_gaps_percent_viterbi,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train_orig1, seqs_train_orig2,
                          seqs_train1, seqs_train2,
                          seqs_class_gaps1, seqs_class_gaps2,
                          avails_gaps1, avails_gaps2,
                          hmms0_size, N, M,
                          hmms0, algorithm='viterbi', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        #
        # gluing
        #
        evaluate_training(ps_gluing, dists_gluing, pi_norms_gluing,
                          a_norms_gluing, b_norms_gluing,
                          class_percent_gluing, class_gaps_percent_gluing,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train_orig1, seqs_train_orig2,
                          seqs_train1, seqs_train2,
                          seqs_class_gaps1, seqs_class_gaps2,
                          avails_gaps1, avails_gaps2,
                          hmms0_size, N, M,
                          hmms0, algorithm='gluing', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        #
        # mode imputation
        #
        evaluate_training(ps_mode, dists_mode, pi_norms_mode,
                          a_norms_mode, b_norms_mode,
                          class_percent_mode, class_gaps_percent_mode,
                          step, n_of_launch, start_time,
                          hmm1, hmm2, seqs_train_orig1, seqs_train_orig2,
                          seqs_train1, seqs_train2,
                          seqs_class_gaps1, seqs_class_gaps2,
                          avails_gaps1, avails_gaps2,
                          hmms0_size, N, M,
                          hmms0, algorithm='mode', rtol=rtol,
                          max_iter=max_iter, avails1=avails1, avails2=avails2,
                          verbose=is_verbose)
        step += 1
#
# get the average values
#
class_percents_best = np.full(len(gaps_range),
                              fill_value=(class_percent_best / n_of_launches))

ps_true = np.full(len(gaps_range), fill_value=(p_true / n_of_launches))

ps_marg /= n_of_launches
pi_norms_marg /= n_of_launches
a_norms_marg /= n_of_launches
b_norms_marg /= n_of_launches
class_percent_marg /= n_of_launches
dists_marg /= n_of_launches
class_gaps_percent_marg /= n_of_launches

ps_viterbi /= n_of_launches
pi_norms_viterbi /= n_of_launches
a_norms_viterbi /= n_of_launches
b_norms_viterbi /= n_of_launches
class_percent_viterbi /= n_of_launches
dists_viterbi /= n_of_launches
class_gaps_percent_viterbi /= n_of_launches

ps_gluing /= n_of_launches
pi_norms_gluing /= n_of_launches
a_norms_gluing /= n_of_launches
b_norms_gluing /= n_of_launches
class_percent_gluing /= n_of_launches
dists_gluing /= n_of_launches
class_gaps_percent_gluing /= n_of_launches

ps_mode /= n_of_launches
pi_norms_mode /= n_of_launches
a_norms_mode /= n_of_launches
b_norms_mode /= n_of_launches
class_percent_mode /= n_of_launches
dists_mode /= n_of_launches
class_gaps_percent_mode /= n_of_launches

#
# plot everything
#
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font', **font)
mpl.rc('font', size=12)
plt.figure(figsize=(1920/96, 1000/96), dpi=96)

suptitle = u"Исследование алгоритмов обучения СММ по посл-тям с пропусками "\
           u"Число скрытых состояний N = {}, число символов M = {}, "\
           u"число обучающих последовательностей K = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число начальных приближений = {}, "\
           u"относительная невязка для останова алгоритма rtol = {}, "\
           u"максимальное число итераций iter = {}, " \
           u"число запусков эксперимента n_of_launches = {} \n"\
           .format(N, M, K, T, hmms0_size, rtol, max_iter, n_of_launches)
plt.suptitle(suptitle)

ax1 = plt.subplot(221)
plt.ylabel(u"Логарифм правдоподобия")
plt.xlabel(u"Процент пропусков")

line1 = plt.plot(xs, ps_marg, '-', label=u"Маргинализация")
line2 = plt.plot(xs, ps_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3 = plt.plot(xs, ps_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4 = plt.plot(xs, ps_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
line5=plt.plot(xs, ps_true, '-', label=u"Истинная СММ без пропусков")

#ax2 = plt.subplot(322)
#plt.ylabel(r"$||\Pi - \Pi^*||$")
#plt.xlabel(u"Процент пропусков")
#line1=plt.plot(xs, pi_norms_marg, '-', label=u"Маргинализация")
#line2=plt.plot(xs, pi_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
#line3=plt.plot(xs, pi_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#line4=plt.plot(xs, pi_norms_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
#
#ax3 = plt.subplot(323, sharex=ax1)
#plt.ylabel(r"$||A - A^*||$")
#plt.xlabel(u"Процент пропусков")
#line1=plt.plot(xs, a_norms_marg, '-', label=u"Маргинализация")
#line2=plt.plot(xs, a_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
#line3=plt.plot(xs, a_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#line4=plt.plot(xs, a_norms_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
#
#ax4 = plt.subplot(324, sharex=ax1)
#plt.ylabel(r"$||B - B^*||$")
#plt.xlabel(u"Процент пропусков")
#line1=plt.plot(xs, b_norms_marg, '-', label=u"Маргинализация")
#line2=plt.plot(xs, b_norms_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
#line3=plt.plot(xs, b_norms_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#line4=plt.plot(xs, b_norms_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")

ax5 = plt.subplot(222, sharex=ax1)
plt.ylabel(r"$D_s(\lambda, \lambda^*)$")
plt.xlabel(u"Процент пропусков")
line1 = plt.plot(xs, dists_marg, '-', label=u"Маргинализация")
line2 = plt.plot(xs, dists_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3 = plt.plot(xs, dists_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4 = plt.plot(xs, dists_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")

ax6 = plt.subplot(223, sharex=ax1)
plt.ylabel(u"Верно распознанные, %")
plt.xlabel(u"Процент пропусков")
line1 = plt.plot(xs, class_percent_marg, '-', label=u"Маргинализация")
line2 = plt.plot(xs, class_percent_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3 = plt.plot(xs, class_percent_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4 = plt.plot(xs, class_percent_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
line5 = plt.plot(xs, class_percents_best, '-', dash_capstyle='round',  lw=2.0, label=u"Истинные модели")

ax7 = plt.subplot(224, sharex=ax1)
plt.ylabel(u"Верно распознанные, %")
plt.xlabel(u"Процент пропусков")
line1 = plt.plot(xs, class_gaps_percent_marg, '-', label=u"Маргинализация")
line2 = plt.plot(xs, class_gaps_percent_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3 = plt.plot(xs, class_gaps_percent_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4 = plt.plot(xs, class_gaps_percent_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
line5 = plt.plot(xs, class_percents_best, '-', dash_capstyle='round',  lw=2.0, label=u"Истинные модели")

plt.figlegend((line1[0], line2[0], line3[0], line4[0], line5[0]), 
              (u"Маргинализация",u"Склеивание",u"Витерби", u"Мода", u"Истинные модели"),
              loc = 'center right')

plt.show()

plt.savefig(filename+".png")

to_file = np.asarray([xs, ps_true, ps_marg, ps_gluing, ps_viterbi, ps_mode,
                      pi_norms_marg, pi_norms_gluing, pi_norms_viterbi, pi_norms_mode,
                      a_norms_marg, a_norms_gluing, a_norms_viterbi, a_norms_mode,
                      b_norms_marg, b_norms_gluing, b_norms_viterbi, b_norms_mode,
                      dists_marg, dists_gluing, dists_viterbi, dists_mode,
                      class_percent_marg, class_percent_gluing, class_percent_viterbi, class_percent_mode,
                      class_gaps_percent_marg, class_gaps_percent_gluing, class_gaps_percent_viterbi, class_gaps_percent_mode,
                      class_percents_best])

np.savetxt(filename+".csv", to_file.T, delimiter=';',
           header="xs;ps_true;ps_marg;ps_gluing;ps_viterbi;ps_mode;"
                  "pi_norms_marg;pi_norms_gluing;pi_norms_viterbi;pi_norms_mode;"
                  "a_norms_marg;a_norms_gluing; a_norms_viterbi;a_norms_mode;"
                  "b_norms_marg;b_norms_gluing; b_norms_viterbi;b_norms_mode;"
                  "dists_marg;dists_gluing;dists_viterbi;dists_mode;"
                  "class_percent_marg; class_percent_gluing; class_percent_viterbi; class_percent_mode;"
                  "class_gaps_percent_marg, class_gaps_percent_gluing, class_gaps_percent_viterbi, class_gaps_percent_mode,"
                  "class_percents_best")

print("--- {} minutes ---".format((time.time()-start_time) / 60))


#xs, ps_true, ps_marg, ps_gluing, ps_viterbi, ps_mode,\
#    pi_norms_marg, pi_norms_gluing, pi_norms_viterbi, pi_norms_mode,\
#    a_norms_marg, a_norms_gluing, a_norms_viterbi, a_norms_mode,\
#    b_norms_marg, b_norms_gluing, b_norms_viterbi, b_norms_mode,\
#    dists_marg, dists_gluing, dists_viterbi, dists_mode,\
#    class_percent_marg, class_percent_gluing, class_percent_viterbi,\
#    class_percent_mode,
#    class_gaps_percent_marg, class_gaps_percent_gluing, class_gaps_percent_viterbi, class_gaps_percent_mode,
#    class_percents_best = \
#    np.loadtxt(filename+".csv", delimiter=';', unpack=True)


