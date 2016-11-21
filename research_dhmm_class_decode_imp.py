# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
import DiscreteHMM as dhmm
import copy
import StandardImputationMethods as imp

start_time = time.time()

dA = 0.4
T = 100
K = 100
n_of_launches = 100
is_gaps_places_different = True
is_verbose = False
out_dir = "out"
filename = "dhmm_dA{}_T{}_K{}_x{}".format(dA, T, K, n_of_launches)
# filename = os.path.join(os.path.dirname(__file__), out_dir, filename)

gaps_range = list(range(0, 100, 10))

#
# HMM
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


def evaluate_classification(class_percent, step, hmm1, hmm2,
                            seqs1, seqs2, avails1=None, avails2=None,
                            algorithm='marginalization',  n_neighbours=10,
                            verbose=False):
    """ Code to evaluate hmm training performance on given sequences
    n_neighbours works only with ‘mode’ algorithm
    """
    print("algorithm = {}".format(algorithm))
    class_res1 = dhmm.classify_seqs(seqs1, [hmm1, hmm2], avails=avails1,
                                    algorithm=algorithm)
    class_res2 = dhmm.classify_seqs(seqs2, [hmm1, hmm2], avails=avails2,
                                    algorithm=algorithm)
    percent = 100.0 * (class_res1.count(0) + class_res2.count(1)) / (2.0 * K)
    class_percent[step] += percent
    print("Correctly classified {} %".format(percent))


def evaluate_decoding_imputation(decoded_percents, imputed_percents, step,
                                 hmm: dhmm.DHMM, seqs, states_list_orig,
                                 seqs_orig, avails, algorithm='viterbi',
                                 n_neighbours=10, verbose=False):
    """ Code to evaluate hmm decoding  and imputation performance on
    given sequences. n_neighbours works only with ‘mode’ algorithm
    """
    assert algorithm in ['viterbi', 'mode']
    print("algorithm = {}".format(algorithm))
    if algorithm == 'viterbi':
        # decode
        states_list_decoded = hmm.decode_viterbi(seqs, avails)
        # then impute
        seqs_imp = hmm.impute_by_states(seqs, avails, states_list_decoded)
    if algorithm == 'mode':
        # impute
        seqs_imp, avails_imp = \
            imp.impute_by_n_neighbours(seqs, avails, n_neighbours,
                                       method='mode', n_of_symbols=hmm._m)
        seqs_imp = imp.impute_by_whole_seq(seqs_imp, avails_imp,
                                           method='mode', n_of_symbols=hmm._m)
        # then decode
        states_list_decoded = hmm.decode_viterbi(seqs_imp)
    # compare decoded states
    compare = np.equal(states_list_orig, states_list_decoded)
    percent = 100.0 * np.count_nonzero(compare) / compare.size
    decoded_percents[step] += percent
    print("Correctly decoded {} %".format(percent))
    # compare imputed sequences
    # TODO: compare only those observations which were missing
    compare = np.equal(seqs_orig, seqs_imp)
    percent = 100.0 * np.count_nonzero(compare) / compare.size
    imputed_percents[step] += percent
    print("Correctly imputed {} %".format(percent))


def make_missing_values(seqs_orig, to_dissapears, n_of_gaps):
    avails = [np.full_like(seqs_orig[i], True, dtype=np.bool)
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
class_percents_mode = np.zeros_like(class_percents_marg)

decoded_percents_viterbi = np.zeros_like(class_percents_marg)
decoded_percents_mode = np.zeros_like(class_percents_marg)

imputed_percents_viterbi = np.zeros_like(class_percents_marg)
imputed_percents_mode = np.zeros_like(class_percents_marg)

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
#        # classification using mode imputation
#        #
#        evaluate_classification(class_percents_mode, step, hmm1, hmm2,
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm='mode', verbose=is_verbose)
        #
        # decoding and imputation using enhanced viterbi
        #
        evaluate_decoding_imputation(decoded_percents_viterbi,
                                     imputed_percents_viterbi, step, hmm1,
                                     seqs1, states_list1, seqs_orig1, avails1,
                                     algorithm='viterbi')
        #
        # decoding and imputation using mode imputation
        #
        evaluate_decoding_imputation(decoded_percents_mode,
                                     imputed_percents_mode, step, hmm1,
                                     seqs1, states_list1, seqs_orig1, avails1,
                                     algorithm='mode')

        print("--- {} minutes ---".format((time.time()-start_time) / 60))
        print()
        step += 1
#
# get the average values
#
class_percents_marg /= n_of_launches
class_percents_viterbi /= n_of_launches
class_percents_gluing /= n_of_launches
class_percents_mode /= n_of_launches

decoded_percents_viterbi /= n_of_launches
decoded_percents_mode /= n_of_launches

imputed_percents_viterbi /= n_of_launches
imputed_percents_mode /= n_of_launches

#
# plot everything
#
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font', **font)
mpl.rc('font', size=12)
# plt.figure(figsize=(1920/96, 1000/96), dpi=96)

plt.figure(1)
suptitle = u"Исследование алгоритмов классификации последовательностей с пропусками "\
           u"Число скрытых состояний N = {}, число символов M = {}, "\
           u"число обучающих последовательностей K = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число запусков эксперимента n_of_launches = {} \n"\
           .format(N, M, K, T, n_of_launches)
plt.suptitle(suptitle)
plt.ylabel(u"Верно распознанные, %")
plt.xlabel(u"Процент пропусков")
line1=plt.plot(xs, class_percents_marg, '-', label=u"Маргинализация")
line2=plt.plot(xs, class_percents_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, class_percents_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4=plt.plot(xs, class_percents_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
plt.legend()
plt.show()
plt.savefig(out_dir + "/class_" + filename + ".png")

plt.figure(2)
suptitle = u"Исследование алгоритмов декодирования последовательностей с пропусками "\
           u"Число скрытых состояний N = {}, число символов M = {}, "\
           u"число обучающих последовательностей K = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число запусков эксперимента n_of_launches = {} \n"\
           .format(N, M, K, T, n_of_launches)
plt.suptitle(suptitle)
plt.ylabel(u"Верно декодированные состояния, %")
plt.xlabel(u"Процент пропусков")
plt.plot(xs, decoded_percents_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
plt.plot(xs, decoded_percents_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
plt.legend()
plt.show()
plt.savefig(out_dir + "/decode_" + filename + ".png")

plt.figure(3)
suptitle = u"Исследование алгоритмов восстановленных последовательностей с пропусками "\
           u"Число скрытых состояний N = {}, число символов M = {}, "\
           u"число обучающих последовательностей K = {}, "\
           u"длина обучающих последовательностей T = {}, \n"\
           u"число запусков эксперимента n_of_launches = {} \n"\
           .format(N, M, K, T, n_of_launches)
plt.suptitle(suptitle)
plt.ylabel(u"Верно восстановленные наблюдения, %")
plt.xlabel(u"Процент пропусков")
plt.plot(xs, imputed_percents_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
plt.plot(xs, imputed_percents_mode, '-.', dash_capstyle='round', lw=2.0, label=u"Мода")
plt.legend()
plt.show()
plt.savefig(out_dir + "/impute_" + filename + ".png")


to_file = np.asarray([xs, class_percents_marg, class_percents_gluing,
                      class_percents_viterbi, class_percents_mode,
                      decoded_percents_viterbi, decoded_percents_mode,
                      imputed_percents_viterbi, imputed_percents_mode])
np.savetxt(out_dir + "/class_decode_impute_" + filename + ".csv", to_file.T, delimiter=';',
           header="xs;class_percents_marg;class_percents_gluing;"
                  "class_percents_viterbi;class_percents_mode;"
                  "decoded_percents_viterbi;decoded_percents_mode;"
                  "imputed_percents_viterbi;imputed_percents_mode")

print("--- {} minutes ---".format((time.time()-start_time) / 60))

filename = "out/dhmm_classification_dA0.4_T100_K100_x100"
xs, class_percents_marg, class_percents_gluing, class_percents_viterbi, class_percents_mode\
    = np.loadtxt(filename+".csv", delimiter=';', unpack=True)
