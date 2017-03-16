# -*- coding: utf-8 -*-
"""
Performing validation to find out the best SVM hypermarameters
"""

import numpy as np
from sklearn import svm
import time
import winsound
import GaussianHMM as ghmm
from research import make_missing_values
from research import gen_gaps_positions
from research import make_svm_training_seqs
import HMM

# validation settings
number_of_trials = 100000   # randomized trials of SVM hypermarameters
kernels=['poly', 'rbf']
C_range = (-3., 10.)
gamma_range = (-11., 3.)
coef0_range = (-100.0, 100.0)
degree_range = (2, 6)
# derivatives calculation settings
wrt = ['a']
# missing data settings
n_of_gaps_train_hmm = 0
n_of_gaps_train_svm = (n_of_gaps_train_hmm, 91)
n_of_gaps_class = (n_of_gaps_train_hmm, 91)
is_gaps_places_different = True
seed_gen_train_gaps = 222
seed_gen_class_gaps = 2
# hmm training settings
rtol = 1e-4
max_iter = 10000
hmms0_size = 5
seed_training = 565
# sequence generation seeds
seed_gen_train_seqs = 565
seed_gen_class_seqs = 777
# size of training and classification samples
T = 100
K = 100
# hmm parameters settings
dA = 0.1
sig_val = 0.1

#
# hmm1
#
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
N, M, Z = mu.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm1_orig = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

#
# hmm 2
#
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
N, M, Z = mu.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm2_orig = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

#
# Generate train sequences
#
train_seqs_orig1, _ = hmm1_orig.generate_sequences(K, T, seed=seed_gen_train_seqs)
train_seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=seed_gen_train_seqs)
#
# generate gaps posititons in HMM training sequences
#
np.random.seed(seed_gen_train_gaps)
train_hmm_to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
train_hmm_to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)
train_hmm_seqs1, train_hmm_avails1 = make_missing_values(train_seqs_orig1, train_hmm_to_dissapears1,
                                                 n_of_gaps_train_hmm)
train_hmm_seqs2, train_hmm_avails2 = make_missing_values(train_seqs_orig2, train_hmm_to_dissapears2,
                                                 n_of_gaps_train_hmm)

#
# generate classification sequences
#
class_seqs_orig1, _ = hmm1_orig.generate_sequences(K, T, seed=seed_gen_class_seqs)
class_seqs_orig2, _ = hmm2_orig.generate_sequences(K, T, seed=seed_gen_class_seqs)
#
# generate gaps in classification sequences
#
np.random.seed(seed_gen_class_gaps)
class_to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
class_to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)
class_seqs1, class_avails1 = make_missing_values(class_seqs_orig1, class_to_dissapears1,
                                                 n_of_gaps_class)
class_seqs2, class_avails2 = make_missing_values(class_seqs_orig2, class_to_dissapears2,
                                                 n_of_gaps_class)

#
# Train HMMs
#
print("training hmms")

# apply baumwelch to train hmms
np.random.seed(seed_training)
hmm1, _, iter1, n_of_best1 = ghmm.train_best_hmm_baumwelch(
                                 train_hmm_seqs1, hmms0_size, N, M, Z,
                                 train_hmm_avails1, rtol, max_iter
                             )
np.random.seed(seed_training)
hmm2, _, iter2, n_of_best2 = ghmm.train_best_hmm_baumwelch(
                                 train_hmm_seqs2, hmms0_size, N, M, Z,
                                 train_hmm_avails2, rtol, max_iter
                             )
print("training completed, iter={}/{}\nhmm1={}\nhmm2={}".format(iter1, iter2, hmm1, hmm2))
distance = HMM.calc_symmetric_distance(hmm1, hmm2, 100, 200)
print("distance between hmms = {}".format(distance))
winsound.Beep(1500, 500)

#
# generate gaps posititons in SVM training sequences
#
np.random.seed(seed_gen_train_gaps)
train_svm_to_dissapears1 = train_hmm_to_dissapears1
train_svm_to_dissapears2 = train_hmm_to_dissapears2
#train_svm_seqs1, train_svm_avails1 = make_missing_values(train_seqs_orig1,
#                                                         train_svm_to_dissapears1,
#                                                         n_of_gaps_train_svm)
#train_svm_seqs2, train_svm_avails2 = make_missing_values(train_seqs_orig2,
#                                                         train_svm_to_dissapears2,
#                                                         n_of_gaps_train_svm)
train_svm_seqs1, train_svm_avails1 = make_svm_training_seqs(train_seqs_orig1,
                                                            train_svm_to_dissapears1,
                                                            range(0, 100, 10))
train_svm_seqs2, train_svm_avails2 = make_svm_training_seqs(train_seqs_orig2,
                                                            train_svm_to_dissapears2,
                                                            range(0, 100, 10))
        
        
        

#
# Precalculate derivatives of training and classification data
#
print("calc derivs for train")
X, y = ghmm._form_train_data_for_SVM([hmm1, hmm2],
                                     [train_svm_seqs1, train_svm_seqs2],
                                     [train_svm_avails1, train_svm_avails2],
                                     wrt)
print("calc derivs for class")
X_class1 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs1, class_avails1, wrt)
X_class2 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs2, class_avails2, wrt)


#
# Random search for hypermarameters of SVM with polynomial kernel
#
# TODO: use RandomizedSearchCV instead!
def SVM_params_generator(n, kernels, C_range, gamma_range, coef0_range, degree_range, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for i in range(n):
        kernel_idx = np.random.randint(0, len(kernels))
        kernel = kernels[kernel_idx]
        C_exp = np.random.uniform(*C_range)
        gamma_exp = np.random.uniform(*gamma_range)
        coef0 = np.random.uniform(*coef0_range)
        degree = np.random.randint(*degree_range)
        yield kernel, 10.0 ** C_exp, 10.0 ** gamma_exp, coef0, degree

cv_grid = []

print("start search")
start = time.time()
for kernel, C, gamma, coef0, degree in SVM_params_generator(
                                           number_of_trials,
                                           kernels,
                                           C_range, gamma_range,
                                           coef0_range, degree_range,
                                           seed=1
                                       ):
    svm_params = svm.SVC(C=C, kernel=kernel, degree=degree,
                         gamma=gamma, coef0=coef0, cache_size=500,
                         max_iter=10000)
    clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2],
                                            [train_svm_seqs1, train_svm_seqs2],
                                            svm_params,
                                            [train_svm_avails1, train_svm_avails2],
                                            X=X, y=y, wrt=wrt)
    svm_predictions1 = ghmm.classify_seqs_svm(class_seqs1, [hmm1, hmm2], clf,
                                              scaler, class_avails1, wrt, X=X_class1)
    svm_predictions2 = ghmm.classify_seqs_svm(class_seqs2, [hmm1, hmm2], clf,
                                              scaler, class_avails2, wrt, X=X_class2)
    perc_svm = (np.count_nonzero(svm_predictions1 == 0) +
                np.count_nonzero(svm_predictions2 == 1)) / (2.0*K) * 100.0
    
    cv_grid.append([kernels.index(kernel), C, gamma, coef0, degree, perc_svm])
    #print(C, gamma, coef0, degree, perc_svm)
    if len(cv_grid) % 1000 == 0:
        print("{:.0f} s elapsed, {:.0f}% completed".format(time.time() - start, 100.0 * len(cv_grid) / number_of_trials))

#
# sort and present the best SVM params
#
array_cv_grid_poly = np.array(cv_grid)
array_cv_grid_poly = array_cv_grid_poly[array_cv_grid_poly[:, 5].argsort()]
print(array_cv_grid_poly[-5:])
print("{:.0f} s passed".format(time.time() - start))
print()
print("best percent")
print(array_cv_grid_poly[-1, 5])
a = array_cv_grid_poly[-1]
print("copy/paste")
print("C={}, kernel='{}', degree={},\ngamma={}, coef0={},"\
      .format(a[1], kernels[int(a[0])], int(a[4]), a[2], a[3]))
print()
winsound.Beep(1000, 1000)

#
# ML classification
#
print("n_of_gaps_class = {}".format(n_of_gaps_class))
#for algorithm in ['marginalization', 'viterbi', 'viterbi_advanced1',
#                  'viterbi_advanced2']:
for algorithm in ['marginalization']:
    print("predicting using likelihood and {}".format(algorithm))
    predictions1 = ghmm.classify_seqs_mlc(class_seqs1, [hmm1, hmm2], class_avails1,
                                          algorithm_gaps=algorithm)
    predictions2 = ghmm.classify_seqs_mlc(class_seqs2, [hmm1, hmm2], class_avails2,
                                          algorithm_gaps=algorithm)
    perc = (np.count_nonzero(predictions1 == 0) +
            np.count_nonzero(predictions2 == 1)) / (2.0*K) * 100.0
    print("Likelihood correct: {}%".format(perc))

winsound.Beep(1000, 3000)




#
# Aldready validated hyperparams
#

# optimal params for true hmms, dA=0.1, wrt='a', n_of_gaps_class=0, n_of_gaps_train = 0
svm_params = svm.SVC(C=7.54540673e-03, kernel='poly', degree=4,
                             gamma=1.28341964e-05, coef0=8.57725599e+01,
                             cache_size=500,
                             max_iter=100000)

# optimal params for true hmms, dA=0.2, wrt='a', n_of_gaps_class=0, n_of_gaps_train = 0
svm_params = svm.SVC(C=11512.946532910306, kernel='poly', degree=5,
                     gamma=9.3719596513688283e-07, coef0=72.199616550878062,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms, dA=0.1, wrt='a', n_of_gaps_class=0, n_of_gaps_train = 0
svm_params = svm.SVC(C=46572.945582839573, kernel='poly', degree=3,
                     gamma=1.6980743332669287e-06, coef0=-50.584281134189666,
                     cache_size=500,
                     max_iter=100000)
# optimal params for trained hmms, dA=0.1, wrt='a', n_of_gaps_class=0, n_of_gaps_train = 10
svm_params = svm.SVC(C=4664688.4796896093, kernel='poly', degree=3,
                     gamma=2.3830295670023817e-08, coef0=-99.337077164302599,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms, dA=0.1, wrt='a',n_of_gaps_class=0, n_of_gaps_train = 0, gmm init full
svm_params = svm.SVC(C=9825241.7140532695, kernel='poly', degree=3,
                     gamma=0.00027821452329615385, coef0=56.351156526706404,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms, dA=0.1, wrt='a',n_of_gaps_class=0, n_of_gaps_train = 0, gmm init diag
svm_params = svm.SVC(C=4664688.4796896093, kernel='poly', degree=3,
                     gamma=2.3830295670023817e-08, coef0=-99.337077164302599,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms, dA=0.1, wrt='a',n_of_gaps_class=0, n_of_gaps_train = 0, gmm init diag, 93.5%
svm_params = svm.SVC(C=9219170863.8300514, kernel='poly', degree=3,
                     gamma=6.6305504742301958e-06, coef0=-96.037731666652192,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms, dA=0.1, wrt='a',n_of_gaps_class=0, n_of_gaps_train_hmm = 0, gmm init diag, 93.0%
# n_of_gaps_train_svm = (0, 90)
svm_params = svm.SVC(
C=462669131.32926476, kernel='poly', degree=5,
gamma=3.619516310745099e-07, coef0=57.8647613117499,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms, dA=0.1, wrt='a', n_of_gaps_class=(0, 90), 
# n_of_gaps_train_hmm = 0, n_of_gaps_train_svm = (0, 90), gmm init diag, 80.5%
svm_params = svm.SVC(
C=3457488.0477916673, kernel='rbf', degree=3,
gamma=5.269878287581289e-08, coef0=-10.104309103960844,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms (25000 trials), dA=0.1, wrt='a', n_of_gaps_train_hmm = 10, 
# n_of_gaps_train_svm = (10, 90), n_of_gaps_class=(10, 90), gmm init diag, 82.5%
svm_params = svm.SVC(
C=3.8882434315566554, kernel='poly', degree=5,
gamma=2.827030299171524e-07, coef0=-20.931734991756116,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms (125000 trials), dA=0.1, wrt='a', n_of_gaps_train_hmm = 10, 
# n_of_gaps_train_svm = (10, 90), n_of_gaps_class=(10, 90), gmm init diag, 83%
svm_params = svm.SVC(
C=229593457.69973943, kernel='poly', degree=3,
gamma=3.741738964859175e-09, coef0=-63.183091664197754,
                     cache_size=500,
                     max_iter=100000)

# optimal params for trained hmms (100000 trials), dA=0.1, wrt='a', n_of_gaps_train_hmm = 0, 
# n_of_gaps_train_svm = make_svm_training_seqs(0, 90), n_of_gaps_class=(0, 90), gmm init diag, 80%
svm_params = svm.SVC(
C=52131480.46723968, kernel='poly', degree=5,
gamma=5.745457655305097e-09, coef0=25.530080740735258,
                     cache_size=500,
                     max_iter=100000)