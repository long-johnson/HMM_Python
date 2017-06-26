# -*- coding: utf-8 -*-

import warnings
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import GaussianHMM as ghmm
import winsound
import sklearn.svm
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.preprocessing
import os
from itertools import product
from research import make_missing_values
from research import gen_gaps_positions
from research import evaluate_classification
from research import evaluate_decoding_imputation
from research import make_svm_training_seqs
import HMM

"""
Comparing different sequence classification techniques
"""
        
#
# Experiment parameters
#
# experiment settings
n_of_launches = 50
is_verbose = False
# derivatives calculation settings
wrt = ['a', 'tau']
# missing data settings
#n_of_gaps_train_hmm = 90

n_of_gaps_train_svm = "make_svm_training_seqs({},100,10)".format(n_of_gaps_train_hmm)
#n_of_gaps_train_svm = n_of_gaps_train_hmm
#n_of_gaps_train_svm = (n_of_gaps_train_hmm, 90)

#class_gaps_range = list(range(0, 100, 10))
class_gaps_range = [n_of_gaps_train_hmm]

is_train_gaps_places_different = True
is_class_gaps_places_different = True
seed_gen_train_gaps = 222
# hmm training settings
rtol = 1e-4
max_iter = 1000
hmms0_size = 5
seed_training = 565
# sequence generation seeds
seed_gen_train_seqs = 565
# size of training and classification samples
T = 100
K = 100
# size of sequences to calc symmetric distances between hmms
T_dist = 500
K_dist = 100
# standard imputation settings
n_neighbours = 10
# SVM hyperparams validation settings
n_trials_svm_random_search = 10000   # randomized trials of SVM hypermarameters
class pow_ten_uniform():
    def __init__(self, a, b):
        self.gen = sp_uniform(a, b)
    def rvs(self):
        return 10.0 ** self.gen.rvs()
cv_n_folds = 4
kernels=['poly', 'rbf']
C_range = (-3., 10.)
gamma_range = (-11., 3.)
coef0_range = (-100.0, 100.0)
degree_range = (2, 6)
cv_random_state = 111
param_distributions = {"C" : pow_ten_uniform(*C_range), "gamma" : pow_ten_uniform(*gamma_range),
                       "coef0" : sp_uniform(*coef0_range), "degree" : sp_randint(*degree_range),
                       "kernel" : ['poly', 'rbf']}
# hmm parameters settings
dA = 0.05
dtau = 0.05
dmu = 0.00
dsig = 0.00
sig_val = 0.1



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

# hmm 2
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
hmm2_orig = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)



#
# Research
#

print("n_of_gaps_train_hmm = {}".format(n_of_gaps_train_hmm))
#
# Train HMMs
#
print("training hmms")
start_time = datetime.datetime.now()
train_seqs_orig1, _ = hmm1_orig.generate_sequences(K, T, seed=seed_gen_train_seqs)
train_seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=seed_gen_train_seqs)

# generate gaps posititons in the training sequences
np.random.seed(seed_gen_train_gaps)
to_dissapears1 = gen_gaps_positions(K, T, is_train_gaps_places_different)
to_dissapears2 = gen_gaps_positions(K, T, is_train_gaps_places_different)
# mark some elements as missing
# hmm 1
train_seqs1, train_avails1 = make_missing_values(train_seqs_orig1, to_dissapears1,
                                                 n_of_gaps_train_hmm)
# hmm 2
train_seqs2, train_avails2 = make_missing_values(train_seqs_orig2, to_dissapears2,
                                                 n_of_gaps_train_hmm)
# apply baumwelch to train hmms
np.random.seed(seed_training)
hmm1, _, iter1, n_of_best1 = ghmm.train_best_hmm_baumwelch(
                                 train_seqs1, hmms0_size, N, M, Z, avails=train_avails1,
                                 hmms0=None, rtol=rtol, max_iter=max_iter
                             )
np.random.seed(seed_training)
hmm2, _, iter2, n_of_best2 = ghmm.train_best_hmm_baumwelch(
                                 train_seqs2, hmms0_size, N, M, Z, avails=train_avails2,
                                 hmms0=None, rtol=rtol, max_iter=max_iter
                             )
print("hmm training completed, iter={}/{}".format(iter1, iter2))
if is_verbose:
    print("hmms:\nhmm1={}\nhmm2={}".format(hmm1, hmm2))
distance_bw_hmm1_hmm2 = HMM.calc_symmetric_distance(hmm1, hmm2, T_dist, K_dist)
print("distance between hmms = {}".format(distance_bw_hmm1_hmm2))
print("---Elapsed: {} ---".format((datetime.datetime.now() - start_time) / 60))

#
# estimate the best possible accuracy
#
seqs_class1, _ = hmm1.generate_sequences(K, T, seed=seed_gen_train_seqs+1)
seqs_class2, _ = hmm2.generate_sequences(K, T, seed=seed_gen_train_seqs+1)
pred1 = ghmm.classify_seqs_mlc(seqs_class1, [hmm1, hmm2])
pred2 = ghmm.classify_seqs_mlc(seqs_class2, [hmm1, hmm2])
percent = (np.count_nonzero(pred1 == 0) + np.count_nonzero(pred2 == 1)) / (2.0*K) * 100.0
print("Best accuracy is {} %".format(percent))
print()

#
# Prepare training sequences for SVM classifier
#
print("Preparing training sequences for SVM classifier")
start_time = datetime.datetime.now()
if n_of_gaps_train_svm == (n_of_gaps_train_hmm, 90):
    # mark some elements as missing with varying percent of gaps between sequences
    train_svm_seqs1, train_svm_avails1 = make_missing_values(train_seqs_orig1, to_dissapears1,
                                                             n_of_gaps_train_svm)
    train_svm_seqs2, train_svm_avails2 = make_missing_values(train_seqs_orig2, to_dissapears2,
                                                             n_of_gaps_train_svm)

elif n_of_gaps_train_svm == "make_svm_training_seqs({},100,10)".format(n_of_gaps_train_hmm):
    # SVM train seqs are formed using make_svm_training_seqs procedure
    train_svm_seqs1, train_svm_avails1 = make_svm_training_seqs(train_seqs_orig1,
                                                            to_dissapears1,
                                                            range(n_of_gaps_train_hmm, 100, 10))
    train_svm_seqs2, train_svm_avails2 = make_svm_training_seqs(train_seqs_orig2,
                                                            to_dissapears2,
                                                            range(n_of_gaps_train_hmm, 100, 10))
elif n_of_gaps_train_svm == n_of_gaps_train_hmm:
    # SVM train seqs are the same as HMM train seqs
    train_svm_seqs1 = train_seqs1
    train_svm_seqs2 = train_seqs2
    train_svm_avails1 = train_avails1
    train_svm_avails2 = train_avails2
else:
    raise Exception("Wrong n_of_gaps_train_svm!")
# precalc derivatives here
X, y = ghmm._form_train_data_for_SVM([hmm1, hmm2],
                                     [train_svm_seqs1, train_svm_seqs2],
                                     [train_svm_avails1, train_svm_avails2],
                                     wrt)
scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)
print("---Elapsed: {} ---".format((datetime.datetime.now() - start_time) / 60))

#
# Optimize SVM hyperparams and train SVM classifier
#
print("Performing cross-validation of SVM params")
start_time = datetime.datetime.now()
clf = sklearn.svm.SVC(cache_size=500, max_iter=10000)
kfold = sklearn.cross_validation.KFold(len(X), n_folds=cv_n_folds, shuffle=True, random_state=cv_random_state)
clf = sklearn.grid_search.RandomizedSearchCV(clf, param_distributions, n_iter=n_trials_svm_random_search,
                                             cv=kfold, random_state=cv_random_state)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    clf.fit(X, y)
print("best score on SVM model on training set: {}".format(clf.best_score_))
print("---Elapsed: {} ---".format((datetime.datetime.now() - start_time) / 60))


#
# Save experiment params to file
#
def save_vars_to_file2(f, names, vals):
    for name, val in zip(names, vals):
        f.write("{} = {}\n".format(name, val))
str_cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = "out/research_ghmm_class_decode_imp_{}/".format(str_cur_time)
os.mkdir(out_dir)
f = open(out_dir+"experiment_params.txt", "w")
save_vars_to_file2(
    f,
    ["n_of_gaps_train_hmm", "n_of_launches", "wrt", "dA", "dtau", "dmu", "dsig", "sig_val",
    "n_of_gaps_train_svm", "class_gaps_range",
     "is_train_gaps_places_different", "is_class_gaps_places_different",
     "seed_gen_train_gaps", "rtol", "max_iter", "hmms0_size", "seed_training",
     "seed_gen_train_seqs", "T", "K", "T_dist", "K_dist", "n_neighbours", "n_trials_svm_random_search",
     "cv_n_folds", "kernels", "C_range", "gamma_range", "coef0_range", "degree_range",
     "cv_random_state", "clf.best_score_",
     "hmm1_orig", "hmm2_orig",
     "hmm1", "hmm2", "iter1", "iter2", "n_of_best1", "n_of_best2",
     "distance_bw_hmm1_hmm2"],
    [n_of_gaps_train_hmm, n_of_launches, wrt, dA, dtau, dmu, dsig, sig_val,
     n_of_gaps_train_svm, class_gaps_range,
     is_train_gaps_places_different, is_class_gaps_places_different,
     seed_gen_train_gaps, rtol, max_iter, hmms0_size, seed_training,
     seed_gen_train_seqs, T, K, T_dist, K_dist, n_neighbours, n_trials_svm_random_search,
     cv_n_folds, kernels, C_range, gamma_range, coef0_range, degree_range,
     cv_random_state, clf.best_score_,
     hmm1_orig, hmm2_orig,
     hmm1, hmm2, iter1, iter2, n_of_best1, n_of_best2,
     distance_bw_hmm1_hmm2]
)
f.close()


#
# initialize variables to store experiment data
#
xs = np.array(class_gaps_range, dtype=np.int)

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
print("Starting the experiment")
start_time = datetime.datetime.now()
for n_of_launch in range(n_of_launches):
    print("n_of_launch = {}".format(n_of_launch))
    # generate sequences to classify
    seqs_orig1, states_list1 = hmm1_orig.generate_sequences(K, T, seed=n_of_launch)
    seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=n_of_launch)

    # generate gaps posititons in the sequences
    np.random.seed(n_of_launch)
    to_dissapears1 = gen_gaps_positions(K, T, is_class_gaps_places_different)
    to_dissapears2 = gen_gaps_positions(K, T, is_class_gaps_places_different)

    # the experiment
    step = 0
    for n_of_gaps in class_gaps_range:
        if is_verbose:
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
        if is_verbose:
            print("---Elapsed: {} ---".format((datetime.datetime.now() - start_time) / 60))
            print()
        step += 1
    print("---Elapsed: {} ---".format((datetime.datetime.now() - start_time) / 60))
    print()
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

winsound.Beep(1000, 3000)

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
suptitle = u"Класс. неполн. послед."\
           u"число/длина обучающих посл-тей K={}/T={},"\
           u"# запусков = {}\n# пропусков в обучающих СММ/SVM={}/{}"\
           .format(K, T, n_of_launches, n_of_gaps_train_hmm, n_of_gaps_train_svm)
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
plt.savefig(out_dir + "classification.png")
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
#plt.savefig(out_dir + "decoding.png")
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
#plt.savefig(out_dir + "imputation.png")

to_file = np.asarray([xs, class_percents_mlc_marg, class_percents_mlc_viterbi,
                      class_percents_mlc_gluing, class_percents_mlc_mean,
                      class_percents_mlc_viterbi1, class_percents_mlc_viterbi2,
                      class_percents_svm_marg, class_percents_svm_viterbi,
                      class_percents_svm_gluing, class_percents_svm_mean,
                      decoded_percents_viterbi, decoded_percents_mean,
                      imputed_sum_squares_diff_viterbi, imputed_sum_squares_diff_mean])
np.savetxt(out_dir + "class_decode_impute.csv", to_file.T, delimiter=',',
           header="xs,class_percents_mlc_marg,class_percents_mlc_viterbi,"
                  "class_percents_mlc_gluing,class_percents_mlc_mean,"
                  "class_percents_mlc_viterbi1,class_percents_mlc_viterbi2,"
                  "class_percents_svm_marg,class_percents_svm_viterbi,"
                  "class_percents_svm_gluing,class_percents_svm_mean,"
                  "decoded_percents_viterbi,decoded_percents_mean,"
                  "imputed_sum_squares_diff_viterbi,imputed_sum_squares_diff_mean")


#filename = "out/dhmm_classification_dA0.4_T100_K100_x100"
#xs, class_percents_marg, class_percents_gluing, class_percents_viterbi, class_percents_mean\
#    = np.loadtxt(filename+".csv", delimiter=';', unpack=True)

