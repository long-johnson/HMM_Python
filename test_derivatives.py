# -*- coding: utf-8 -*-
"""
Testing derivatives calculations

"""

import numpy as np
from sklearn import svm
import time
import winsound
import copy
import GaussianHMM as ghmm

sig_val = 0.1
T = 100
K = 100
dA = 0.1
is_gaps_places_different = True
n_of_gaps = 20

# hmm1
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
N, M, Z = mu.shape
sig = np.empty((N,M,Z,Z))
for n in range(N):
    for m in range(M):
        sig[n,m,:,:] = np.eye(Z) * sig_val
hmm2 = ghmm.GHMM(N, M, Z, mu, sig, pi=pi, a=a, tau=tau)

train_seqs_orig1, _ = hmm1.generate_sequences(K, T, seed=1)
train_seqs_orig2, _ = hmm2.generate_sequences(K, T, seed=1)
class_seqs_orig1, _ = hmm1.generate_sequences(K, T, seed=2)
class_seqs_orig2, _ = hmm2.generate_sequences(K, T, seed=2)

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

class_to_dissapears1 = gen_gaps_positions(K, T, is_gaps_places_different)
class_to_dissapears2 = gen_gaps_positions(K, T, is_gaps_places_different)

class_seqs1, class_avails1 = make_missing_values(class_seqs_orig1, class_to_dissapears1,
                                                 n_of_gaps)
# hmm 2
class_seqs2, class_avails2 = make_missing_values(class_seqs_orig2, class_to_dissapears2,
                                                 n_of_gaps)


#
# derivatves calculation
#

#seqs, _ = hmm1.generate_sequences(K, T, seed=1)

#derivs = hmm1.calc_derivatives(seqs)

#print("derivs")
#print(derivs)


#
# SVM classifer
#
#print("training SVM")
#train_seqs1, _ = hmm1.generate_sequences(K, T, seed=1)
#train_seqs2, _ = hmm2.generate_sequences(K, T, seed=1)
#svm_params = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
#                     probability=False, shrinking=1, tol=1e-3, cache_size=500,
#                     verbose=True, )
#clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2], [train_seqs1, train_seqs2],
#                                        svm_params)
#
#print("generating class seqs")
#class_seqs1, _ = hmm1.generate_sequences(K, T, seed=2)
#class_seqs2, _ = hmm2.generate_sequences(K, T, seed=2)
#
#print("predicting  using SVM")
#svm_predictions1 = ghmm.classify_seqs_svm(class_seqs1, [hmm1, hmm2], clf, scaler)
#svm_predictions2 = ghmm.classify_seqs_svm(class_seqs2, [hmm1, hmm2], clf, scaler)
#perc_svm = (np.count_nonzero(svm_predictions1 == 0) +
#            np.count_nonzero(svm_predictions2 == 1)) / (2.0*K) * 100.0
#print("SVM correct: {}%".format(perc_svm))
#

print("n_of_gaps = {}".format(n_of_gaps))
for algorithm in ['marginalization', 'viterbi', 'viterbi_advanced1',
                  'viterbi_advanced2']:
    print("predicting using likelihood and {}".format(algorithm))
    predictions1 = ghmm.classify_seqs(class_seqs1, [hmm1, hmm2], avails=class_avails1,
                                      algorithm=algorithm)
    predictions2 = ghmm.classify_seqs(class_seqs2, [hmm1, hmm2], avails=class_avails1,
                                      algorithm=algorithm)
    perc = (np.count_nonzero(predictions1 == 0) +
            np.count_nonzero(predictions2 == 1)) / (2.0*K) * 100.0
    print("Likelihood correct: {}%".format(perc))

    
#
# SVM RBF hypermarameters random search
#
#def SVM_RBF_generator(n, C_range, gamma_range, seed=None):
#    if seed is not None:
#        np.random.seed(seed)
#    for i in range(n):
#        C_exp = np.random.uniform(*C_range)
#        gamma_exp = np.random.uniform(*gamma_range)
#        yield 10.0 ** C_exp, 10.0 ** gamma_exp
#
#
#X, y = ghmm._form_train_data_for_SVM([hmm1, hmm2], [train_seqs1, train_seqs2])
#X_class1 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs1)
#X_class2 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs2)
#cv_grid_rbf = []
#
#start = time.time()
#for C, gamma in SVM_RBF_generator(n=10000, C_range=(5., 6.), gamma_range=(-8., -7.),
#                                  seed=6):
#    svm_params = svm.SVC(C=C, kernel='rbf', gamma=gamma,
#                         max_iter=100000, verbose=False)
#    clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2],
#                                            [train_seqs1, train_seqs2],
#                                            svm_params, X=X, y=y)
#    svm_predictions1 = ghmm.classify_seqs_svm(class_seqs1, [hmm1, hmm2], clf,
#                                              scaler, X=X_class1)
#    svm_predictions2 = ghmm.classify_seqs_svm(class_seqs2, [hmm1, hmm2], clf,
#                                              scaler, X=X_class2)
#    perc_svm = (np.count_nonzero(svm_predictions1 == 0) +
#                np.count_nonzero(svm_predictions2 == 1)) / (2.0*K) * 100.0
#    cv_grid_rbf.append([C, gamma, perc_svm])
#    print(C, gamma, perc_svm)
#    print("{:.0f} s passed".format(time.time() - start))
#    
#array_cv_grid_rbf = np.array(cv_grid_rbf)
#array_cv_grid_rbf = array_cv_grid_rbf[array_cv_grid_rbf[:, 2].argsort()]
#print(array_cv_grid_rbf[-5:])
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(np.log10(array_cv_grid_rbf[:, 0]),
#           np.log10(array_cv_grid_rbf[:, 1]),
#           array_cv_grid_rbf[:, 2])
#ax.set_xlabel('C')
#ax.set_ylabel('gamma')
#ax.set_zlabel('acc, %')
#plt.show()

#
# Params for wrt='a'
# svm_params = svm.SVC(C=7.54540673e-03, kernel='poly', degree=4,
#                     gamma=1.28341964e-05, coef0=8.57725599e+01, cache_size=500,
#                     max_iter=100000)


#wrt = None
##
## SVM with polynomial kernel
##
#start = time.time()
#print("calc derivs for train")
#X, y = ghmm._form_train_data_for_SVM([hmm1, hmm2],
#                                     [train_seqs_orig1, train_seqs_orig2],
#                                     wrt=wrt)
#print("{:.0f} s passed".format(time.time() - start))
#print("calc derivs for class1")
#start = time.time()
#X_class1 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs1,
#                                         avails=class_avails1,
#                                         wrt=wrt)
#print("calc derivs for class2")
#X_class2 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs2,
#                                         avails=class_avails2,
#                                         wrt=wrt)
#print("{:.0f} s passed".format(time.time() - start))
#
#print("Training svm and classifying 1 time")
#import GaussianHMM as ghmm
#start = time.time()
#svm_params = svm.SVC(C=7.54540673e-03, kernel='poly', degree=4,
#                     gamma=1.28341964e-05, coef0=8.57725599e+01, cache_size=500,
#                     max_iter=100000)
#clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2],
#                                        [train_seqs_orig1, train_seqs_orig2],
#                                        svm_params, X=X, y=y)
#svm_predictions1 = ghmm.classify_seqs_svm(class_seqs1, [hmm1, hmm2], clf,
#                                          scaler, avails=class_avails1,
#                                          X=X_class1)
#svm_predictions2 = ghmm.classify_seqs_svm(class_seqs2, [hmm1, hmm2], clf,
#                                          scaler, avails=class_avails2,
#                                          X=X_class2)
#perc_svm = (np.count_nonzero(svm_predictions1 == 0) +
#            np.count_nonzero(svm_predictions2 == 1)) / (2.0*K) * 100.0
#print(perc_svm)
#print("{:.0f} s passed".format(time.time() - start))


#
# SVM polynomial kernel hypermarameters random search
#
#def SVM_poly_generator(n, C_range, gamma_range, coef0_range, degree_range, seed=None):
#    if seed is not None:
#        np.random.seed(seed)
#    for i in range(n):
#        C_exp = np.random.uniform(*C_range)
#        gamma_exp = np.random.uniform(*gamma_range)
#        coef0 = np.random.uniform(*coef0_range)
#        degree = np.random.randint(*degree_range)
#        yield 10.0 ** C_exp, 10.0 ** gamma_exp, coef0, degree
#
#
#print("calc derivs for train")
#X, y = ghmm._form_train_data_for_SVM([hmm1, hmm2],
#                                     [train_seqs_orig1, train_seqs_orig2],
#                                     wrt="a")
#print("calc derivs for class")
#X_class1 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs1, wrt="a")
#X_class2 = ghmm._form_class_data_for_SVM([hmm1, hmm2], class_seqs2, wrt="a")
#cv_grid_poly = []
#winsound.Beep(500, 1000)
#
#print("start search")
#start = time.time()
#for C, gamma, coef0, degree in SVM_poly_generator(n=1000, C_range=(-3., 10.),
#                                                  gamma_range=(-11., 3.),
#                                                  coef0_range=(-100.0, 100.0),
#                                                  degree_range=(3, 5),
#                                                  seed=1):
#    svm_params = svm.SVC(C=C, kernel='poly', degree=degree,
#                         gamma=gamma, coef0=coef0, cache_size=500,
#                         max_iter=1000000)
#    clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2],
#                                            [train_seqs_orig1, train_seqs_orig2],
#                                            svm_params, X=X, y=y)
#    svm_predictions1 = ghmm.classify_seqs_svm(class_seqs1, [hmm1, hmm2], clf,
#                                              scaler, X=X_class1)
#    svm_predictions2 = ghmm.classify_seqs_svm(class_seqs2, [hmm1, hmm2], clf,
#                                              scaler, X=X_class2)
#    perc_svm = (np.count_nonzero(svm_predictions1 == 0) +
#                np.count_nonzero(svm_predictions2 == 1)) / (2.0*K) * 100.0
#    cv_grid_poly.append([C, gamma, coef0, degree, perc_svm])
#    print(C, gamma, coef0, degree, perc_svm)
#    print("{:.0f} s passed".format(time.time() - start))
#    
#array_cv_grid_poly = np.array(cv_grid_poly)
#array_cv_grid_poly = array_cv_grid_poly[array_cv_grid_poly[:, 4].argsort()]
#print(array_cv_grid_poly[-5:])
#
#winsound.Beep(1000, 3000)