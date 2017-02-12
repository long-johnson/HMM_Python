# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import GaussianHMM as ghmm
import copy
from sklearn import svm
import StandardImputationMethods as imp
from gaps_generation import make_missing_values
from gaps_generation import gen_gaps_positions

start_time = time.time()

n_of_launches = 5
dA = 0.1
sig_val = 0.1
rtol = 1e-4
wrt = ['a']
n_of_gaps_train = 10
T = 100
K = 100
hmms0_size = 5
max_iter = 10000

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
hmm1_orig = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

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
hmm2_orig = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)


def evaluate_classification(class_percent, step, hmms,
                            seqs1, seqs2, avails1=None, avails2=None,
                            algorithm_class='mlc',
                            algorithm_gaps='marginalization', n_neighbours=10,
                            clf=None, scaler=None, wrt=None,
                            verbose=False):
    """ Code to evaluate hmm training performance on given sequences
    algorithm_class: {'MLC' - maximum likelihood classifier, 'svm'}
    algorithm_gaps: {'marginalization', 'viterbi', 'viterbi_advanced1',
                     'viterbi_advanced2', 'mean'}
    n_neighbours works only with ‘mean’ algorithm_gaps
    clf and scaler work only with 'svm' algorithm
    'viterbi_advanced1' and 'viterbi_advanced2'
    """
    print("algorithm_class = {}, algorithm_gaps = {}".format(algorithm_class,
                                                             algorithm_gaps))
    if algorithm_class == 'mlc':
        pred1 = ghmm.classify_seqs_mlc(seqs1, hmms, avails=avails1,
                                       algorithm_gaps=algorithm_gaps)
        pred2 = ghmm.classify_seqs_mlc(seqs2, hmms, avails=avails2,
                                       algorithm_gaps=algorithm_gaps)
    if algorithm_class == 'svm':
        pred1 = ghmm.classify_seqs_svm(seqs1, hmms, clf, scaler, avails=avails1,
                                       algorithm_gaps=algorithm_gaps, wrt=wrt)
        pred2 = ghmm.classify_seqs_svm(seqs2, hmms, clf, scaler, avails=avails2,
                                       algorithm_gaps=algorithm_gaps, wrt=wrt)
    percent = (np.count_nonzero(pred1 == 0) +
               np.count_nonzero(pred2 == 1)) / (2.0*K) * 100.0
    class_percent[step] += percent
    print("Correctly classified {} %".format(percent))


def evaluate_decoding_imputation(decoded_percents, imputed_sum_squares_diff, step,
                                 hmm: ghmm.GHMM, seqs, states_list_orig,
                                 seqs_orig, avails, algorithm_gaps='viterbi',
                                 n_neighbours=10, verbose=False):
    """ Code to evaluate hmm decoding  and imputation performance on
    given sequences. n_neighbours works only with ‘mean’ algorithm
    
    sum_squares_diff - average (of K sequences) of average (of length of each sequence)
        squared difference between original sequence and imputed
    """
    assert algorithm_gaps in ['viterbi', 'mean']
    print("algorithm = {}".format(algorithm_gaps))
    if algorithm_gaps == 'viterbi':
        # decode
        states_list_decoded = hmm.decode_viterbi(seqs, avails)
        # then impute
        seqs_imp = hmm.impute_by_states(seqs, avails, states_list_decoded)
    if algorithm_gaps == 'mean':
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

#
# Research
#

#
# Train HMMs
#
print("training hmms")
train_seqs_orig1, _ = hmm1_orig.generate_sequences(K, T, seed=565)
train_seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=565)

# generate gaps posititons in the taining sequences
np.random.seed(111)
to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)
# mark some elements as missing
# hmm 1
train_seqs1, train_avails1 = make_missing_values(train_seqs_orig1, to_dissapears1,
                                                 n_of_gaps_train)
# hmm 2
train_seqs2, train_avails2 = make_missing_values(train_seqs_orig2, to_dissapears2,
                                                 n_of_gaps_train)
# apply baumwelch to train hmms
np.random.seed(565)
hmm1, _, iter1, n_of_best1 = \
    ghmm.train_best_hmm_baumwelch(train_seqs1, hmms0_size, N, M, Z, avails=train_avails1,
                                  hmms0=None, rtol=rtol, max_iter=max_iter)
np.random.seed(565)
hmm2, _, iter2, n_of_best2 = \
    ghmm.train_best_hmm_baumwelch(train_seqs2, hmms0_size, N, M, Z, avails=train_avails2,
                                  hmms0=None, rtol=rtol, max_iter=max_iter)
print("training completed, iter={}/{}\nhmm1={}\nhmm2={}".format(iter1, iter2, hmm1, hmm2))


# train SVM classifier

print("training SVM classifier")
svm_params = svm.SVC(C=46572.945582839573, kernel='poly', degree=3,
                     gamma=1.6980743332669287e-06, coef0=-50.584281134189666,
                     cache_size=500,
                     max_iter=100000)
clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2],
                                        [train_seqs1, train_seqs2],
                                        svm_params, [train_avails1, train_avails2],
                                        wrt=wrt)

#
# initialize variables to store experiment data
#
xs = np.array(gaps_range, dtype=np.int)

class_percents_mlc_marg = np.zeros_like(xs, dtype=np.float)
class_percents_mlc_viterbi = np.zeros_like(class_percents_mlc_marg)
class_percents_mlc_viterbi1 = np.zeros_like(class_percents_mlc_marg)
class_percents_mlc_viterbi2 = np.zeros_like(class_percents_mlc_marg)
class_percents_mlc_gluing = np.zeros_like(class_percents_mlc_marg)
class_percents_mlc_mean = np.zeros_like(class_percents_mlc_marg)

class_percents_svm_marg = np.zeros_like(class_percents_mlc_marg)
class_percents_svm_viterbi = np.zeros_like(class_percents_mlc_marg)
class_percents_svm_gluing = np.zeros_like(class_percents_mlc_marg)
class_percents_svm_mean = np.zeros_like(class_percents_mlc_marg)

decoded_percents_viterbi = np.zeros_like(class_percents_mlc_marg)
decoded_percents_mean = np.zeros_like(class_percents_mlc_marg)

imputed_sum_squares_diff_viterbi = np.zeros_like(class_percents_mlc_marg)
imputed_sum_squares_diff_mean = np.zeros_like(class_percents_mlc_marg)


# make several launches on different seeds to get the average result
for n_of_launch in range(n_of_launches):
    print("n_of_launch = {}".format(n_of_launch))
    # generate sequences to classify
    seqs_orig1, states_list1 = hmm1_orig.generate_sequences(K, T, seed=n_of_launch)
    seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=n_of_launch)

    # generate gaps posititons in the sequences
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
        #
        # mlc classification using marginalization
        #
        evaluate_classification(class_percents_mlc_marg, step, [hmm1, hmm2],
                                seqs1, seqs2, avails1=avails1, avails2=avails2,
                                algorithm_class='mlc',
                                algorithm_gaps='marginalization',
                                verbose=is_verbose)
#        #
#        # mlc classification using viterbi
#        #
#        evaluate_classification(class_percents_mlc_viterbi, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='mlc',
#                                algorithm_gaps='viterbi', verbose=is_verbose)
#        #
#        # mlc classification using gluing
#        #
#        evaluate_classification(class_percents_mlc_gluing, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='mlc',
#                                algorithm_gaps='gluing', verbose=is_verbose)
#        #
#        # mlc classification using mean imputation
#        #
#        evaluate_classification(class_percents_mlc_mean, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='mlc',
#                                algorithm_gaps='mean', verbose=is_verbose,
#                                n_neighbours=n_neighbours)
#        #
#        # mlc classification using advanced viterbi v.1 imputation
#        #
#        evaluate_classification(class_percents_mlc_viterbi1, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='mlc',
#                                algorithm_gaps='viterbi_advanced1', verbose=is_verbose)
#        #
#        # mlc classification using advanced viterbi v.2 imputation
#        #
#        evaluate_classification(class_percents_mlc_viterbi2, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='mlc',
#                                algorithm_gaps='viterbi_advanced2', verbose=is_verbose)


        #
        # svm classification using marginalization
        #
        evaluate_classification(class_percents_svm_marg, step, [hmm1, hmm2],
                                seqs1, seqs2, avails1=avails1, avails2=avails2,
                                algorithm_class='svm',
                                algorithm_gaps='marginalization',
                                clf=clf, scaler=scaler, wrt=wrt,
                                verbose=is_verbose)
#        #
#        # svm classification using viterbi
#        #
#        evaluate_classification(class_percents_svm_viterbi, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='svm',
#                                algorithm_gaps='viterbi',
#                                clf=clf, scaler=scaler, wrt=wrt,
#                                verbose=is_verbose)
#        #
#        # svm classification using gluing
#        #
#        evaluate_classification(class_percents_svm_gluing, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='svm',
#                                algorithm_gaps='gluing',
#                                clf=clf, scaler=scaler, wrt=wrt,
#                                verbose=is_verbose)
#        #
#        # svm classification using mean imputation
#        #
#        evaluate_classification(class_percents_svm_mean, step, [hmm1, hmm2],
#                                seqs1, seqs2, avails1=avails1, avails2=avails2,
#                                algorithm_class='svm',
#                                algorithm_gaps='mean',
#                                clf=clf, scaler=scaler, wrt=wrt,
#                                verbose=is_verbose,
#                                n_neighbours=n_neighbours)


        #
        # decoding and imputation using enhanced viterbi algorithm
        #
#        evaluate_decoding_imputation(decoded_percents_viterbi,
#                                     imputed_sum_squares_diff_viterbi, step, hmm1,
#                                     seqs1, states_list1, seqs_orig1, avails1,
#                                     algorithm_gaps='viterbi')
#        #
#        # decoding and imputation using mean imputation
#        #
#        evaluate_decoding_imputation(decoded_percents_mean,
#                                     imputed_sum_squares_diff_mean, step, hmm1,
#                                     seqs1, states_list1, seqs_orig1, avails1,
#                                     algorithm_gaps='mean', n_neighbours=n_neighbours)

        print("--- {} minutes ---".format((time.time()-start_time) / 60))
        print()
        step += 1
#
# get the average values
#
class_percents_mlc_marg /= n_of_launches
class_percents_mlc_viterbi /= n_of_launches
class_percents_mlc_gluing /= n_of_launches
class_percents_mlc_mean /= n_of_launches
class_percents_mlc_viterbi1 /= n_of_launches
class_percents_mlc_viterbi2 /= n_of_launches

class_percents_svm_marg /= n_of_launches
class_percents_svm_viterbi /= n_of_launches
class_percents_svm_gluing /= n_of_launches
class_percents_svm_mean /= n_of_launches

decoded_percents_viterbi /= n_of_launches
decoded_percents_mean /= n_of_launches

imputed_sum_squares_diff_viterbi /= n_of_launches
imputed_sum_squares_diff_mean /= n_of_launches

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
plt.plot(xs, class_percents_mlc_marg, '-', label=u"ММП, Маргинализация")
#plt.plot(xs, class_percents_mlc_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"ММП, Склеивание")
#plt.plot(xs, class_percents_mlc_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"ММП, Витерби")
#plt.plot(xs, class_percents_mlc_mean, '-.', dash_capstyle='round', lw=2.0, label=u"ММП, Среднее")
#plt.plot(xs, class_percents_mlc_viterbi1, ':', dash_capstyle='round', lw=2.0, label=u"ММП, Витерби, v.1")
#plt.plot(xs, class_percents_mlc_viterbi2, ':', dash_capstyle='round', lw=2.0, label=u"ММП, Витерби, v.2")
plt.plot(xs, class_percents_svm_marg, '-', label=u"SVM, Маргинализация")
#plt.plot(xs, class_percents_svm_gluing, '--',  dash_capstyle='round',  lw=2.0, label=u"SVM, Склеивание")
#plt.plot(xs, class_percents_svm_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"SVM, Витерби")
#plt.plot(xs, class_percents_svm_mean, '-.', dash_capstyle='round', lw=2.0, label=u"SVM, Среднее")
plt.legend()
plt.show()
plt.savefig(out_dir + "/class_" + filename + ".png")

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
#plt.plot(xs, decoded_percents_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")
#plt.legend()
#plt.show()
#plt.savefig(out_dir + "/decode_" + filename + ".png")
#
#plt.figure(3)
#suptitle = u"Исследование алгоритмов восстановленных последовательностей с пропусками "\
#           u"Число скрытых состояний N = {}, число символов M = {}, "\
#           u"число обучающих последовательностей K = {}, "\
#           u"длина обучающих последовательностей T = {}, \n"\
#           u"число запусков эксперимента n_of_launches = {} \n"\
#           .format(N, M, K, T, n_of_launches)
#plt.suptitle(suptitle)
#plt.ylabel(u"Разница м/у исходными и восстановленными наблюдениями")
#plt.xlabel(u"Процент пропусков")
#plt.plot(xs, imputed_sum_squares_diff_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#plt.plot(xs, imputed_sum_squares_diff_mean, '-.', dash_capstyle='round', lw=2.0, label=u"Среднее")
#plt.legend()
#plt.show()
#plt.savefig(out_dir + "/impute_" + filename + ".png")


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
