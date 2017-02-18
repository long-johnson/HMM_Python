# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import GaussianHMM as ghmm
import winsound
from sklearn import svm
from research import make_missing_values
from research import gen_gaps_positions
from research import evaluate_classification
from research import evaluate_decoding_imputation
import HMM


start_time = time.time()

# missing data settings
n_of_gaps_train_hmm = 0
n_of_gaps_train_svm = (n_of_gaps_train_hmm, 91)
gaps_range = list(range(0, 100, 10))
is_gaps_places_different = True
# hmms settings
dA = 0.1
sig_val = 0.1
# training settings
rtol = 1e-4
max_iter = 10000
hmms0_size = 5
# derivatives calculation settings
wrt = ['a']
# sequence generation settings
T = 100
K = 100
# recognition settings
n_neighbours = 10
# experiment settings
n_of_launches = 10
is_verbose = False
out_dir = "out/"
# svm classifier settings
# optimal params for trained hmms, dA=0.1, wrt='a', n_of_gaps_class=(0, 90), 
# n_of_gaps_train_hmm = 0, n_of_gaps_train_svm = (0, 90), gmm init diag, 80.5%
svm_params = svm.SVC(
C=3457488.0477916673, kernel='rbf', degree=3,
gamma=5.269878287581289e-08, coef0=-10.104309103960844,
                     cache_size=500,
                     max_iter=100000)

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

filename = "ghmm_dA={}_sigval={}_N={}_M={}_Z={}_K={}_T={}_n_of_gaps_train={}"\
    "_launches={}_diagcov={}_n_neighbours={}_n_of_gaps_train_svm={}_gmminit"\
    .format(dA, sig_val, N, M, Z, K, T, n_of_gaps_train_hmm, n_of_launches,
            ghmm.is_cov_diagonal, n_neighbours, n_of_gaps_train_svm)

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

#
# Research
#
print("n_of_gaps_train_hmm = {}".format(n_of_gaps_train_hmm))
#
# Train HMMs
#
print("training hmms")
train_seqs_orig1, _ = hmm1_orig.generate_sequences(K, T, seed=565)
train_seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=565)

# generate gaps posititons in the training sequences
np.random.seed(111)
to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)
# mark some elements as missing
# hmm 1
train_seqs1, train_avails1 = make_missing_values(train_seqs_orig1, to_dissapears1,
                                                 n_of_gaps_train_hmm)
# hmm 2
train_seqs2, train_avails2 = make_missing_values(train_seqs_orig2, to_dissapears2,
                                                 n_of_gaps_train_hmm)
# apply baumwelch to train hmms
np.random.seed(565)
hmm1, _, iter1, n_of_best1 = ghmm.train_best_hmm_baumwelch(
                                 train_seqs1, hmms0_size, N, M, Z, avails=train_avails1,
                                 hmms0=None, rtol=rtol, max_iter=max_iter
                             )
np.random.seed(565)
hmm2, _, iter2, n_of_best2 = ghmm.train_best_hmm_baumwelch(
                                 train_seqs2, hmms0_size, N, M, Z, avails=train_avails2,
                                 hmms0=None, rtol=rtol, max_iter=max_iter
                             )
print("training completed, iter={}/{}\nhmm1={}\nhmm2={}".format(iter1, iter2, hmm1, hmm2))
distance = HMM.calc_symmetric_distance(hmm1, hmm2, 100, 200)
print("distance between hmms = {}".format(distance))
winsound.Beep(1000, 1000)

#
# train SVM classifier
#
# mark some elements as missing with varying percent of gaps between sequences
# hmm 1
train_seqs1, train_avails1 = make_missing_values(train_seqs_orig1, to_dissapears1,
                                                 n_of_gaps_train_svm)
# hmm 2
train_seqs2, train_avails2 = make_missing_values(train_seqs_orig2, to_dissapears2,
                                                 n_of_gaps_train_svm)
# train
print("training SVM classifier")
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
           u"Число скрытых состояний N = {}, число символов M = {}, \n"\
           u"число/длина обучающих последовательностей K = {}/T={},"\
           u"число запусков эксперимента n_of_launches = {}, # пропусков в обучающих СММ={}\n"\
           u"# пропусков в обучающих SVM={}\n"\
           .format(N, M, K, T, n_of_launches, n_of_gaps_train_hmm, n_of_gaps_train_svm)
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
plt.close()

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
