# -*- coding: utf-8 -*-

"""
    Testing various features of HMM classifier
"""

import numpy as np

from HiddenMarkovModel import HiddenMarkovModel
from HiddenMarkovModel import generate_discrete_distribution
from HiddenMarkovModel import train_hmm_baumwelch_noscaling
from HiddenMarkovModel import choose_best_hmm_using_bauwelch
# generating
T = 50
N = 2
M = 2
"""Pi = np.array([0.4, 0.6])
A = np.array([
    [0.2, 0.8],
    [0.8, 0.2]
    ])
B = np.array([
    [0.4, 0.6],
    [0.7, 0.3]
    ])
hmm = HiddenMarkovModel(N,M,pi=Pi,a=A,b=B)"""

#hmm = HiddenMarkovModel(N, M)

hmm = HiddenMarkovModel(N, M, seed=563)

seq = hmm.generate_sequence(T, seed=564)
print seq
print np.histogram(seq, bins=[0,1,2,3], normed=True)

# calculating usual likelihood
lhood_noscale, alphas_noscale = hmm.calc_forward_noscaling(seq)
print "Likelihood noscale=" + str(lhood_noscale)

# calculating log-sum-exp likelihood
lhood_logsumexp, alphas_logsumexp = hmm.calc_forward_logsumexp(seq, T)
print "Likelihood log-sum-exp=" + str(lhood_logsumexp)

#absolute difference between usual and log-sum-exp likelihood
#print "absDiff = " + str(np.abs(lhood_noscale-lhood_logsumexp))

#relative difference between usual and log-sum-exp likelihood
#print "relDiff = " + str(np.abs(lhood_noscale-lhood_logsumexp) / lhood_noscale)

print "alphas noscale=" + str(alphas_noscale)

#print "alphas log-sum-exp=" + str(alphas_logsumexp)

# calculate difference between alphas of noscaling and logsumexp:
#print "alphas absdiff = " + str(np.abs(alphas_noscale-alphas_logsumexp))

# calculate difference between alphas of noscaling and logsumexp:
#print "alphas reldiff = " \
#      + str(np.abs(alphas_noscale-alphas_logsumexp) / alphas_noscale)
print
# testing generate_discrete_distribution function
#distr = generate_discrete_distribution(10)
#print distr
#print np.sum(distr)

#
#hmm = HiddenMarkovModel(N, M, seed=236)
print "\nHMM generated params:"
print hmm.pi
print hmm.a
print hmm.b

print 
print "No scaling: " + str(hmm.calc_forward_noscaling(seq)[0])
print "Log-sum-exp: " + str(hmm.calc_forward_logsumexp(seq, T)[0])
loglikelihood, sc_alpha, c = hmm.calc_forward_scaled(seq, T)
print "Scaled: " + str(np.exp(loglikelihood))

# forward variables
print
print "forward variables not scaled"
print hmm.calc_forward_noscaling(seq)[1]
check_sc_alpha = np.empty(shape=(T,N))
for t in range(T):
    check_sc_alpha[t,:] = sc_alpha[t,:] / np.prod(c[:t+1])
print "check scaled forward variables"
print check_sc_alpha

# backaward variables
print
print "bacward variables not scaled"
print hmm.calc_backward_noscaling(seq, T)
sc_beta = hmm.calc_backward_scaled(seq, T, c)
#print sc_beta[:,:]
check_sc_beta = np.empty(shape=(T,N))
for t in reversed(range(T)):
    check_sc_beta[t,:] = sc_beta[t,:] / np.prod(c[t:])
print "check scaled backward variables"
print check_sc_beta

""" training noscaling
"""
print
# initial approximation
hmm0 = HiddenMarkovModel(N, M, seed=11)
hmm_trained = train_hmm_baumwelch_noscaling(seq, hmm0)

print "\nHMM generated params:"
print hmm.pi
print hmm.a
print hmm.b

print "initial approx:"
print hmm0.pi
print hmm0.a
print hmm0.b

print "trained model:"
print hmm_trained.pi
print hmm_trained.a
print hmm_trained.b
print
print "compare likelihoods:"
print "trained: " + str(hmm_trained.calc_forward_noscaling(seq)[0])
print "true: " + str(hmm.calc_forward_noscaling(seq)[0])
print hmm.calc_forward_noscaling(seq)[0] > hmm_trained.calc_forward_noscaling(seq)[0]

# classification check
print
print "classification check"
T=50
hmm1 = HiddenMarkovModel(n=3,m=3,seed=11)
hmm2 = HiddenMarkovModel(n=3,m=3,seed=21)
seq1 = hmm1.generate_sequence(T, seed=10)
seq2 = hmm2.generate_sequence(T, seed=20)
print "1m-1s: " + str(hmm1.calc_forward_noscaling(seq1)[0])
print "1m-2s: " + str(hmm1.calc_forward_noscaling(seq2)[0])
print "2m-1s: " + str(hmm2.calc_forward_noscaling(seq1)[0])
print "2m-2s: " + str(hmm2.calc_forward_noscaling(seq2)[0])


hmm_trained = \
    choose_best_hmm_using_bauwelch(seq, train_hmm_baumwelch_noscaling, 
                                   10, 2, 2, rtol=0.05, max_iter=50
                                   )

print "\nHMM generated params:"
print hmm.pi
print hmm.a
print hmm.b

print "trained model:"
print hmm_trained.pi
print hmm_trained.a
print hmm_trained.b
print
print "compare likelihoods:"
print "trained: " + str(hmm_trained.calc_forward_noscaling(seq)[0])
print "true: " + str(hmm.calc_forward_noscaling(seq)[0])
print hmm.calc_forward_noscaling(seq)[0] > hmm_trained.calc_forward_noscaling(seq)[0]