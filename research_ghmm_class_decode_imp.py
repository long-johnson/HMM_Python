# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import GaussianHMM as ghmm
import copy
import StandardImputationMethods as imp

start_time = time.time()

dA = 0.2
sig_val = 0.1
T = 100
K = 100
n_of_launches = 100
is_gaps_places_different = True
is_verbose = False
out_dir = "out/"
n_neighbours = 10

gaps_range = list(range(0, 100, 10))

#
# HMMs
#
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

filename = "ghmm_dA={}_sigval={}_N={}_M={}_Z={}_K={}_T={}"\
    "_launches={}_diagcov={}_n_neighbours={}"\
    .format(dA, sig_val, N, M, Z, K, T, n_of_launches, ghmm.is_cov_diagonal, n_neighbours)

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


def evaluate_classification(class_percent, step, hmm1, hmm2,
                            seqs1, seqs2, avails1=None, avails2=None,
                            algorithm='marginalization',  n_neighbours=10,
                            verbose=False):
    """ Code to evaluate hmm training performance on given sequences
    n_neighbours works only with ‘mean’ algorithm
    """
    print("algorithm = {}".format(algorithm))
    class_res1 = ghmm.classify_seqs(seqs1, [hmm1, hmm2], avails=avails1,
                                    algorithm=algorithm)
    class_res2 = ghmm.classify_seqs(seqs2, [hmm1, hmm2], avails=avails2,
                                    algorithm=algorithm)
    percent = 100.0 * (class_res1.count(0) + class_res2.count(1)) / (2.0 * K)
    class_percent[step] += percent
    print("Correctly classified {} %".format(percent))


def evaluate_decoding_imputation(decoded_percents, imputed_sum_squares_diff, step,
                                 hmm: ghmm.GHMM, seqs, states_list_orig,
                                 seqs_orig, avails, algorithm='viterbi',
                                 n_neighbours=10, verbose=False):
    """ Code to evaluate hmm decoding  and imputation performance on
    given sequences. n_neighbours works only with ‘mean’ algorithm
    
    sum_squares_diff - average (of K sequences) of average (of length of each sequence)
        squared difference between original sequence and imputed
    """
    assert algorithm in ['viterbi', 'mean']
    print("algorithm = {}".format(algorithm))
    if algorithm == 'viterbi':
        # decode
        states_list_decoded = hmm.decode_viterbi(seqs, avails)
        # then impute
        seqs_imp = hmm.impute_by_states(seqs, avails, states_list_decoded)
    if algorithm == 'mean':
        # impute
        seqs_imp, avails_imp = \
            imp.impute_by_n_neighbours(seqs, avails, n_neighbours,
                                       method='mean')
        seqs_imp = imp.impute_by_whole_seq(seqs_imp, avails_imp,
                                           method='mean')
        # then decode
        states_list_decoded = hmm.decode_viterbi(seqs_imp)
    # compare decoded states
    compare = np.equal(states_list_orig, states_list_decoded)
    percent = 100.0 * np.count_nonzero(compare) / compare.size
    decoded_percents[step] += percent
    print("Correctly decoded {} %".format(percent))
    # compare imputed sequences
    # TODO: compare only those observations which were missing
    sum_squares_diff = 0.0
    K = len(seqs_orig)
    for k in range(K):
        T = len(seqs_orig[k])
        sum_squares_diff += np.sum((seqs_orig[k] - seqs_imp[k]) ** 2) / T
    sum_squares_diff /= K
    imputed_sum_squares_diff[step] += sum_squares_diff
    print("Imputed sum_squares_diff: {}".format(sum_squares_diff))


def make_missing_values(seqs_orig, to_dissapears, n_of_gaps):
    avails = [np.full(len(seqs_orig[i]), True, dtype=np.bool)
              for i in range(K)]
    seqs = copy.deepcopy(seqs_orig)
    for k in range(K):
        avails[k][to_dissapears[k][:n_of_gaps]] = False
        seqs[k][to_dissapears[k][:n_of_gaps]] = -20000
    return seqs, avails


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


#
# initialize variables to store experiment data
#
xs = np.array(gaps_range, dtype=np.int)

class_percents_marg = np.zeros_like(xs, dtype=np.float)
class_percents_viterbi = np.zeros_like(class_percents_marg)
class_percents_gluing = np.zeros_like(class_percents_marg)
class_percents_mean = np.zeros_like(class_percents_marg)

decoded_percents_viterbi = np.zeros_like(class_percents_marg)
decoded_percents_mean = np.zeros_like(class_percents_marg)

imputed_sum_squares_diff_viterbi = np.zeros_like(class_percents_marg)
imputed_sum_squares_diff_mean = np.zeros_like(class_percents_marg)

#
# Research
#
for n_of_launch in range(n_of_launches):
    print("n_of_launch = {}".format(n_of_launch))
    # generate new sequence
    seqs_orig1, states_list1 = hmm1.generate_sequences(K, T, seed=n_of_launch)
    seqs_orig2, _ = hmm2.generate_sequences(K, T, seed=n_of_launch)

    # generate gaps postitons
    np.random.seed(n_of_launch)
    to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
    to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)

    # the experiment
    step = 0
    for n_of_gaps in gaps_range:
        print("n_of_gaps = {}".format(n_of_gaps))
        # mark some elements as missing
        # hmm 1
        seqs1, avails1 = make_missing_values(seqs_orig1, to_dissapears1,
                                             n_of_gaps)
        # hmm 2
        seqs2, avails2 = make_missing_values(seqs_orig2, to_dissapears2,
                                             n_of_gaps)
#        #
#        # classification using marginalization
#        #
#        evaluate_classification(class_percents_marg, step, hmm1, hmm2,
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm='marginalization',
#                                verbose=is_verbose)
#        #
#        # classification using viterbi
#        #
#        evaluate_classification(class_percents_viterbi, step, hmm1, hmm2,
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm='viterbi', verbose=is_verbose)
#        #
#        # classification using gluing
#        #
#        evaluate_classification(class_percents_gluing, step, hmm1, hmm2,
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm='gluing', verbose=is_verbose)
#        #
#        # classification using mean imputation
#        #
#        evaluate_classification(class_percents_mean, step, hmm1, hmm2,
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm='mean', verbose=is_verbose)
        #
        # decoding and imputation using enhanced viterbi
        #
        evaluate_decoding_imputation(decoded_percents_viterbi,
                                     imputed_sum_squares_diff_viterbi, step, hmm1,
                                     seqs1, states_list1, seqs_orig1, avails1,
                                     algorithm='viterbi')
        #
        # decoding and imputation using mean imputation
        #
        evaluate_decoding_imputation(decoded_percents_mean,
                                     imputed_sum_squares_diff_mean, step, hmm1,
                                     seqs1, states_list1, seqs_orig1, avails1,
                                     algorithm='mean', n_neighbours=n_neighbours)

        print("--- {} minutes ---".format((time.time()-start_time) / 60))
        print()
        step += 1
#
# get the average values
#
class_percents_marg /= n_of_launches
class_percents_viterbi /= n_of_launches
class_percents_gluing /= n_of_launches
class_percents_mean /= n_of_launches

decoded_percents_viterbi /= n_of_launches
decoded_percents_mean /= n_of_launches

imputed_sum_squares_diff_viterbi /= n_of_launches
imputed_sum_squares_diff_mean /= n_of_launches

#
# plot everything
#
#mpl.rcdefaults()
#font = {'family': 'Verdana',
#        'weight': 'normal'}
#mpl.rc('font', **font)
#mpl.rc('font', size=12)
# plt.figure(figsize=(1920/96, 1000/96), dpi=96)

#plt.figure(1)
#suptitle = u"Исследование алгоритмов классификации последовательностей с пропусками "\
#           u"Число скрытых состояний N = {}, число символов M = {}, "\
#           u"число обучающих последовательностей K = {}, "\
#           u"длина обучающих последовательностей T = {}, \n"\
#           u"число запусков эксперимента n_of_launches = {} \n"\
#           .format(N, M, K, T, n_of_launches)
#plt.suptitle(suptitle)
#plt.ylabel(u"Верно распознанные, %")
#plt.xlabel(u"Процент пропусков")
#line1=plt.plot(xs, class_percents_marg, '-', label=u"Маргинализация")
#line2=plt.plot(xs, class_percents_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
#line3=plt.plot(xs, class_percents_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#line4=plt.plot(xs, class_percents_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
#plt.legend()
#plt.show()
#plt.savefig(out_dir + "/class_" + filename + ".png")
#
#plt.figure(2)
#suptitle = u"Исследование алгоритмов декодирования последовательностей с пропусками "\
#           u"Число скрытых состояний N = {}, число символов M = {}, "\
#           u"число обучающих последовательностей K = {}, "\
#           u"длина обучающих последовательностей T = {}, \n"\
#           u"число запусков эксперимента n_of_launches = {} \n"\
#           .format(N, M, K, T, n_of_launches)
#plt.suptitle(suptitle)
#plt.ylabel(u"Верно декодированные состояния, %")
#plt.xlabel(u"Процент пропусков")
#plt.plot(xs, decoded_percents_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#plt.plot(xs, decoded_percents_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
#plt.legend()
#plt.show()
#plt.savefig(out_dir + "/decode_" + filename + ".png")

plt.figure(3)
suptitle = u"Исследование алгоритмов восстановленных последовательностей с пропусками "\
           u"Число скрытых состояний N = {}, число символов M = {}, "\
           u"число обучающих последовательностей K = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число запусков эксперимента n_of_launches = {} \n"\
           .format(N, M, K, T, n_of_launches)
plt.suptitle(suptitle)
plt.ylabel(u"Разница м/у исходными и восстановленными наблюдениями")
plt.xlabel(u"Процент пропусков")
plt.plot(xs, imputed_sum_squares_diff_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
plt.plot(xs, imputed_sum_squares_diff_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
plt.legend()
plt.show()
plt.savefig(out_dir + "/impute_" + filename + ".png")


#to_file = np.asarray([xs, class_percents_marg, class_percents_gluing,
#                      class_percents_viterbi, class_percents_mean,
#                      decoded_percents_viterbi, decoded_percents_mean,
#                      imputed_percents_viterbi, imputed_percents_mean])
#np.savetxt(out_dir + "/class_decode_impute_" + filename + ".csv", to_file.T, delimiter=';',
#           header="xs;class_percents_marg;class_percents_gluing;"
#                  "class_percents_viterbi;class_percents_mean;"
#                  "decoded_percents_viterbi;decoded_percents_mean;"
#                  "imputed_percents_viterbi;imputed_percents_mean")
#
#print("--- {} minutes ---".format((time.time()-start_time) / 60))

#filename = "out/dhmm_classification_dA0.4_T100_K100_x100"
#xs, class_percents_marg, class_percents_gluing, class_percents_viterbi, class_percents_mean\
#    = np.loadtxt(filename+".csv", delimiter=';', unpack=True)
