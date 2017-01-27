# -*- coding: utf-8 -*-
"""
Testing derivatives calculations

"""

import numpy as np
from sklearn import svm
import GaussianHMM as ghmm

sig_val = 0.1
T = 100
K = 100
dA = 0.1

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
print("training SVM")
train_seqs1, _ = hmm1.generate_sequences(K, T, seed=1)
train_seqs2, _ = hmm2.generate_sequences(K, T, seed=1)
svm_params = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                     probability=False, shrinking=1, tol=1e-3, cache_size=500,
                     verbose=True, )
clf, scaler = ghmm.train_svm_classifier([hmm1, hmm2], [train_seqs1, train_seqs2],
                                        svm_params)

print("generating class seqs")
class_seqs1, _ = hmm1.generate_sequences(K, T, seed=2)
class_seqs2, _ = hmm2.generate_sequences(K, T, seed=2)

print("predicting  using SVM")
svm_predictions1 = ghmm.classify_seqs_svm(class_seqs1, [hmm1, hmm2], clf, scaler)
svm_predictions2 = ghmm.classify_seqs_svm(class_seqs2, [hmm1, hmm2], clf, scaler)
perc_svm = (np.count_nonzero(svm_predictions1 == 0) +
            np.count_nonzero(svm_predictions2 == 1)) / (2.0*K) * 100.0
print("SVM correct: {}%".format(perc_svm))

print("predicting using likelihood")
predictions1 = ghmm.classify_seqs(class_seqs1, [hmm1, hmm2])
predictions2 = ghmm.classify_seqs(class_seqs2, [hmm1, hmm2])
perc = (np.count_nonzero(predictions1 == 0) +
        np.count_nonzero(predictions2 == 1)) / (2.0*K) * 100.0
print("Likelihood correct: {}%".format(perc))