# -*- coding: utf-8 -*-

"""
    Testing various features of HMM classifier
"""

import numpy as np

from HiddenMarkovModel import HiddenMarkovModel
from HiddenMarkovModel import generate_discrete_distribution

# generating
T = 100
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

seq = hmm.generate_sequence(T, seed=563)
print seq
print np.histogram(seq, bins=[0,1,2,3], normed=True)

# calculating usual likelihood
lhood_noscale, alphas_noscale = hmm.calc_likelihood_noscaling(seq, T)
print "Likelihood noscale=" + str(lhood_noscale)

# calculating log-sum-exp likelihood
lhood_logsumexp, alphas_logsumexp = hmm.calc_likelihood_logsumexp(seq, T)
print "Likelihood log-sum-exp=" + str(lhood_logsumexp)

#absolute difference between usual and log-sum-exp likelihood
print "absDiff = " + str(np.abs(lhood_noscale-lhood_logsumexp))

#relative difference between usual and log-sum-exp likelihood
print "relDiff = " + str(np.abs(lhood_noscale-lhood_logsumexp) / lhood_noscale)

print "alphas noscale=" + str(alphas_noscale)

print "alphas log-sum-exp=" + str(alphas_logsumexp)

# calculate difference between alphas of noscaling and logsumexp:
print "alphas absdiff = " + str(np.abs(alphas_noscale-alphas_logsumexp))

# calculate difference between alphas of noscaling and logsumexp:
print "alphas reldiff = " \
      + str(np.abs(alphas_noscale-alphas_logsumexp) / alphas_noscale)
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
print "No scaling: " + str(hmm.calc_likelihood_noscaling(seq, T)[0])
print "Log-sum-exp: " + str(hmm.calc_likelihood_logsumexp(seq, T)[0])
print "Scaled: " + str(np.exp(hmm.calc_likelihood_scaled(seq, T)[0]))