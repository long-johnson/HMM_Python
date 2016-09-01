# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import time
import DiscreteHMM as dhmm
import StandardImputationMethods as stdimp

start_time = time.time()

dA = 0.1
rtol=1e-5
max_iter=1000
T = 600
K = 50
K_class = 100
hmms0_size = 5
use_predefined_hmms0 = False
number_of_launches = 1
is_gaps_places_different = True
#filename = "ultimate"+"_dA"+str(dA)+"_t"+str(T)+"_k"+str(K)+"_initrand"+\
#    str(hmms0_size*np.logical_not(use_predefined_hmms0))\
#    +"_rtol"+str(rtol)+"_iter"+str(max_iter)+"_x"+str(number_of_launches)
filename = "mode_imp2"
#gaps_range = range(0,T,T/10)
gaps_range = range(0,100,25) + range(100,600,50) + [575] + [590]

# hmm 1
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.2, 0.6],
              [0.8, 0.1, 0.1]])
b = np.array([[0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.1]])
n, m = b.shape
hmm1 = dhmm.DHMM(n, m, pi=pi, a=a, b=b)

# hmm 2
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.1+dA, 0.7-dA, 0.2],
              [0.2, 0.2+dA, 0.6-dA],
              [0.8-dA, 0.1, 0.1+dA]])
b = np.array([[0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.1]])
hmm2 = dhmm.DHMM(n, m, pi=pi, a=a, b=b)

# hmm 0
pi = np.array([1.0, 0.0, 0.0])
a = np.array([[0.6, 0.2, 0.2],
              [0.2, 0.6, 0.2],
              [0.2, 0.2, 0.6]])
b = np.array([[0.2, 0.2, 0.6],
              [0.2, 0.6, 0.2],
              [0.6, 0.2, 0.2]])
hmm0 = dhmm.DHMM(n, m, pi=pi, a=a, b=b)

#
# research
#
if use_predefined_hmms0:
    hmms0 = [hmm0]
else:
    hmms0 = [dhmm.DHMM(n,m,seed=1) for i in range(hmms0_size)]
xs = gaps_range
class_percent_best = np.full(len(gaps_range), 0.0) # if classified by true models
ps1 = np.full(len(gaps_range), 0.0)
a_norms1 = np.full(len(gaps_range), 0.0)
b_norms1 = np.full(len(gaps_range), 0.0)
class_percent = np.full(len(gaps_range), 0.0)
ps_glue1 = np.full(len(gaps_range), 0.0)
a_norms_glue1 = np.full(len(gaps_range), 0.0)
b_norms_glue1 = np.full(len(gaps_range), 0.0)
class_percent_glue = np.full(len(gaps_range), 0.0)
ps_viterbi1 = np.full(len(gaps_range), 0.0)
a_norms_viterbi1 = np.full(len(gaps_range), 0.0)
b_norms_viterbi1 = np.full(len(gaps_range), 0.0)
class_percent_viterbi = np.full(len(gaps_range), 0.0)
for n_of_launch in range(number_of_launches):
    # generate new sequence
    seqs_train1, state_seqs_train1 = hmm1.generate_sequences(K, T, seed=n_of_launch)
    seqs_train2, state_seqs_train2 = hmm2.generate_sequences(K, T, seed=n_of_launch)
    # prepare indices in sequence array to dissapear
    to_dissapears1 = []
    to_dissapears2 = []
    np.random.seed(n_of_launch)
    # generate gaps postitons
    if is_gaps_places_different:
        # gaps are in different places in sequences
        for k in range(K):
            to_dissapear = np.arange(T)
            np.random.shuffle(to_dissapear)
            to_dissapears1.append(to_dissapear)
            np.random.shuffle(to_dissapear)
            to_dissapears2.append(to_dissapear)
    else:
         # gaps are at the same place in sequences
        to_dissapear1 = np.arange(T)
        np.random.shuffle(to_dissapear1)
        to_dissapear2 = np.arange(T)
        np.random.shuffle(to_dissapear2)
        for k in range(K):
            to_dissapears1.append(to_dissapear1)
            to_dissapears2.append(to_dissapear2)
    # generate sequences to classify
    seqs_class1,_ = hmm1.generate_sequences(K_class, T, seed=n_of_launch)
    seqs_class2,_ = hmm2.generate_sequences(K_class, T, seed=n_of_launch)
        
    # the experiment
    step = 0
    for n_of_gaps in gaps_range:
        # mark some elements as missing.
        # array to set missing obserations
        # hmm 1
        avails1 = [np.full_like(seqs_train1[i], True, dtype=np.bool) for i in range(K)]
        train_seqs1 = [np.array(seqs_train1[k]) for k in range(K)]
        for k in range(K):
            avails1[k][to_dissapears1[k][:n_of_gaps]] = False
            train_seqs1[k][to_dissapears1[k][:n_of_gaps]] = -20000
        # hmm 2
        avails2 = [np.full_like(seqs_train2[i], True, dtype=np.bool) for i in range(K)]
        train_seqs2 = [np.array(seqs_train2[k]) for k in range(K)]
        for k in range(K):
            avails2[k][to_dissapears2[k][:n_of_gaps]] = False
            train_seqs2[k][to_dissapears2[k][:n_of_gaps]] = -20000   
            
        # best classification by true models 
        class_res1 = dhmm.classify_seqs(seqs_class1, [hmm1, hmm2])
        class_res2 = dhmm.classify_seqs(seqs_class2, [hmm1, hmm2])
        percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
        class_percent_best[step] += percent
        print "Best percent is " + str(percent) + " %"
        
        # MODE imputation
        hmm_trained1, iter1 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs1, hmms0_size, n, m, hmms0=hmms0,
                                                algorithm='mode', rtol=rtol,
                                                max_iter=max_iter, avails=avails1)
        hmm_trained2, iter2 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs2, hmms0_size, n, m, hmms0=hmms0,
                                                algorithm = 'mode', rtol=rtol,
                                                max_iter=max_iter, avails=avails2)
        if hmm_trained1 is None or hmm_trained2 is None:
            ps1[step] += 0.0
            a_norms1[step] += 2.0
            b_norms1[step] += 2.0
            class_percent[step] += 50.0
        else:
            diff_a1 = np.linalg.norm(hmm_trained1._a-hmm1._a)
            diff_b1 = np.linalg.norm(hmm_trained1._b-hmm1._b)
            diff_a2 = np.linalg.norm(hmm_trained2._a-hmm2._a)
            diff_b2 = np.linalg.norm(hmm_trained2._b-hmm2._b)
            #diff_a = (diff_a1 + diff_a2) / 2.0
            #diff_b = (diff_b1 + diff_b2) / 2.0
            print n_of_gaps
            print "Marginalization"
            print "model1"
            print hmm_trained1._pi
            print hmm_trained1._a
            print hmm_trained1._b
            print "model2"
            print hmm_trained2._pi
            print hmm_trained2._a
            print hmm_trained2._b
            print "loglikelihood = " + str(np.log(hmm_trained1.calc_likelihood_noscale(seqs_train1)))
            print "loglikelihood true = " + str(np.log(hmm1.calc_likelihood_noscale(seqs_train1)))
            print "norm of A diff = " + str(diff_a1) + " / " + str(diff_a2)
            print "norm of B diff = " + str(diff_b1) + " / " + str(diff_b2)
            print "Iterations: " + str(iter1) + " / " + str(iter2)
            ps1[step] += hmm_trained1.calc_likelihood_noscale(seqs_train1)
            a_norms1[step] += diff_a1
            b_norms1[step] += diff_b1
            # classification
            #seqs_class1,_ = hmm1.generate_sequences(K_class, T, seed=n_of_gaps)
            #seqs_class2,_ = hmm2.generate_sequences(K_class, T, seed=n_of_gaps)
            class_res1 = dhmm.classify_seqs(seqs_class1, [hmm_trained1, hmm_trained2])
            class_res2 = dhmm.classify_seqs(seqs_class2, [hmm_trained1, hmm_trained2])
            percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
            class_percent[step] += percent
            print str(percent) + " %"
        """
        # marginalization
        hmm_trained1, iter1 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs1, hmms0_size, n, m, hmms0=hmms0,
                                                algorithm='marginalization', rtol=rtol,
                                                max_iter=max_iter, avails=avails1)
        hmm_trained2, iter2 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs2, hmms0_size, n, m, hmms0=hmms0,
                                                algorithm = 'marginalization', rtol=rtol,
                                                max_iter=max_iter, avails=avails2)
        if hmm_trained1 is None or hmm_trained2 is None:
            ps1[step] += 0.0
            a_norms1[step] += 2.0
            b_norms1[step] += 2.0
            class_percent[step] += 50.0
        else:
            diff_a1 = np.linalg.norm(hmm_trained1._a-hmm1._a)
            diff_b1 = np.linalg.norm(hmm_trained1._b-hmm1._b)
            diff_a2 = np.linalg.norm(hmm_trained2._a-hmm2._a)
            diff_b2 = np.linalg.norm(hmm_trained2._b-hmm2._b)
            #diff_a = (diff_a1 + diff_a2) / 2.0
            #diff_b = (diff_b1 + diff_b2) / 2.0
            print n_of_gaps
            print "Marginalization"
            print "model1"
            print hmm_trained1._pi
            print hmm_trained1._a
            print hmm_trained1._b
            print "model2"
            print hmm_trained2._pi
            print hmm_trained2._a
            print hmm_trained2._b
            print "loglikelihood = " + str(np.log(hmm_trained1.calc_likelihood_noscale(seqs_train1)))
            print "loglikelihood true = " + str(np.log(hmm1.calc_likelihood_noscale(seqs_train1)))
            print "norm of A diff = " + str(diff_a1) + " / " + str(diff_a2)
            print "norm of B diff = " + str(diff_b1) + " / " + str(diff_b2)
            print "Iterations: " + str(iter1) + " / " + str(iter2)
            ps1[step] += hmm_trained1.calc_likelihood_noscale(seqs_train1)
            a_norms1[step] += diff_a1
            b_norms1[step] += diff_b1
            # classification
            #seqs_class1,_ = hmm1.generate_sequences(K_class, T, seed=n_of_gaps)
            #seqs_class2,_ = hmm2.generate_sequences(K_class, T, seed=n_of_gaps)
            class_res1 = dhmm.classify_seqs(seqs_class1, [hmm_trained1, hmm_trained2])
            class_res2 = dhmm.classify_seqs(seqs_class2, [hmm_trained1, hmm_trained2])
            percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
            class_percent[step] += percent
            print str(percent) + " %"
        
        # gluing
        hmm_trained1, iter1 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs1, hmms0_size,  n, m, hmms0=hmms0,
                                                algorithm = 'gluing', rtol=rtol,
                                                max_iter=max_iter, avails=avails1)
        hmm_trained2, iter2 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs2, hmms0_size,  n, m, hmms0=hmms0,
                                                algorithm = 'gluing', rtol=rtol,
                                                max_iter=max_iter, avails=avails2)
        if hmm_trained1 is None or hmm_trained2 is None:
            ps_glue1[step] += 0.00
            a_norms_glue1[step] += 2.0
            b_norms_glue1[step] += 2.0
            class_percent_glue[step] += 50.0
        else:
            diff_a1 = np.linalg.norm(hmm_trained1._a-hmm1._a)
            diff_b1 = np.linalg.norm(hmm_trained1._b-hmm1._b)
            diff_a2 = np.linalg.norm(hmm_trained2._a-hmm2._a)
            diff_b2 = np.linalg.norm(hmm_trained2._b-hmm2._b)
            print n_of_gaps
            print "Gluing"
            print hmm_trained1._pi
            print hmm_trained1._a
            print hmm_trained1._b
            print "model2"
            print hmm_trained2._pi
            print hmm_trained2._a
            print hmm_trained2._b
            print "loglikelihood = " + str(np.log(hmm_trained1.calc_likelihood_noscale(seqs_train1)))
            print "loglikelihood true = " + str(np.log(hmm1.calc_likelihood_noscale(seqs_train1)))
            print "norm of A diff = " + str(diff_a1) + " / " + str(diff_a2)
            print "norm of B diff = " + str(diff_b1) + " / " + str(diff_b2)
            print "Iterations: " + str(iter1) + " / " + str(iter2)
            ps_glue1[step] += hmm_trained1.calc_likelihood_noscale(seqs_train1)
            a_norms_glue1[step] += diff_a1
            b_norms_glue1[step] += diff_b1
            # classification
            #seqs_class1,_ = hmm1.generate_sequences(K_class, T, seed=n_of_gaps)
            #seqs_class2,_ = hmm2.generate_sequences(K_class, T, seed=n_of_gaps)
            class_res1 = dhmm.classify_seqs(seqs_class1, [hmm_trained1, hmm_trained2])
            class_res2 = dhmm.classify_seqs(seqs_class2, [hmm_trained1, hmm_trained2])
            percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
            class_percent_glue[step] += percent
            print str(percent) + " %"
        
        # viterbi
        hmm_trained1, iter1 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs1, hmms0_size,  n, m, hmms0=hmms0,
                                                algorithm = 'viterbi', rtol=rtol,
                                                max_iter=max_iter, avails=avails1)
        hmm_trained2, iter2 = \
            dhmm.choose_best_hmm_using_bauwelch(train_seqs2, hmms0_size,  n, m, hmms0=hmms0,
                                                algorithm = 'viterbi', rtol=rtol,
                                                max_iter=max_iter, avails=avails2)
        if hmm_trained1 is None or hmm_trained2 is None:
            ps_viterbi1[step] += 0.00
            a_norms_viterbi1[step] += 2.0
            b_norms_viterbi1[step] += 2.0
            class_percent_viterbi[step] += 50.0
        else:
            diff_a1 = np.linalg.norm(hmm_trained1._a-hmm1._a)
            diff_b1 = np.linalg.norm(hmm_trained1._b-hmm1._b)
            diff_a2 = np.linalg.norm(hmm_trained2._a-hmm2._a)
            diff_b2 = np.linalg.norm(hmm_trained2._b-hmm2._b)
            print n_of_gaps
            print "Viterbi"
            print hmm_trained1._pi
            print hmm_trained1._a
            print hmm_trained1._b
            print "model2"
            print hmm_trained2._pi
            print hmm_trained2._a
            print hmm_trained2._b
            print "loglikelihood = " + str(np.log(hmm_trained1.calc_likelihood_noscale(seqs_train1)))
            print "loglikelihood true = " + str(np.log(hmm1.calc_likelihood_noscale(seqs_train1)))
            print "norm of A diff = " + str(diff_a1) + " / " + str(diff_a2)
            print "norm of B diff = " + str(diff_b1) + " / " + str(diff_b2)
            print "Iterations: " + str(iter1) + " / " + str(iter2)
            ps_viterbi1[step] += hmm_trained1.calc_likelihood_noscale(seqs_train1)
            a_norms_viterbi1[step] += diff_a1
            b_norms_viterbi1[step] += diff_b1
            # classification
            #seqs_class1,_ = hmm1.generate_sequences(K_class, T, seed=n_of_gaps)
            #seqs_class2,_ = hmm2.generate_sequences(K_class, T, seed=n_of_gaps)
            class_res1 = dhmm.classify_seqs(seqs_class1, [hmm_trained1, hmm_trained2])
            class_res2 = dhmm.classify_seqs(seqs_class2, [hmm_trained1, hmm_trained2])
            percent = 100.0*(class_res1.count(0) + class_res2.count(1)) / (2.0*K_class)
            class_percent_viterbi[step] += percent
            print str(percent) + " %"
            """
        
        print("--- %.1f minutes ---" % ((time.time()-start_time) / 60))
        step += 1

# get the average value
class_percent_best[:] /= number_of_launches
ps1[:] /= number_of_launches
a_norms1[:] /= number_of_launches
b_norms1[:] /= number_of_launches
class_percent[:] /= number_of_launches
ps_glue1[:] /= number_of_launches
a_norms_glue1[:] /= number_of_launches
b_norms_glue1[:] /= number_of_launches
class_percent_glue[:] /= number_of_launches
ps_viterbi1[:] /= number_of_launches
a_norms_viterbi1[:] /= number_of_launches
b_norms_viterbi1[:] /= number_of_launches
class_percent_viterbi[:] /= number_of_launches

# plot all this
mpl.rcdefaults()
font = {'family': 'Verdana',
        'weight': 'normal'}
mpl.rc('font',**font)
mpl.rc('font', size=12)
plt.figure(figsize=(1920/96, 1000/96), dpi=96)

ax1 = plt.subplot(411)
plt.ylabel(u"Логарифм правдоподобия")
#plt.xlabel(u"Число пропусков")
line1=plt.plot(xs, np.log(ps1), '-', label=u"Маргинализация")
line2=plt.plot(xs, np.log(ps_glue1), '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
line3=plt.plot(xs, np.log(ps_viterbi1), ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.legend(loc='lower left')

ax2 = plt.subplot(412, sharex=ax1)
plt.ylabel("||A*-A||")
#plt.xlabel(u"Число пропусков")
plt.plot(xs, a_norms1, '-', label=u"Маргинализация")
plt.plot(xs, a_norms_glue1, '--',  dash_capstyle='round',  lw=2.0,  label=u"Склеивание")
plt.plot(xs, a_norms_viterbi1, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#plt.setp(ax2.get_xticklabels(), visible=False)
#plt.legend(loc='upper left')

ax3 = plt.subplot(413, sharex=ax1)
plt.ylabel("||B*-B||")
#plt.xlabel(u"Число пропусков")
plt.plot(xs, b_norms1, '-', label=u"Маргинализация")
plt.plot(xs, b_norms_glue1, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
plt.plot(xs, b_norms_viterbi1, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
#plt.setp(ax3.get_xticklabels(), visible=False)
#plt.legend(loc='upper left')

plt.subplot(414, sharex=ax1)
plt.ylabel(u"Верно распознанные, %")
plt.xlabel(u"Число пропусков")
plt.plot(xs, class_percent, '-', label=u"Маргинализация")
plt.plot(xs, class_percent_glue, '--',  dash_capstyle='round',  lw=2.0, label=u"Склеивание")
plt.plot(xs, class_percent_viterbi, ':', dash_capstyle='round', lw=2.0, label=u"Витерби")
line4 = plt.plot(xs, class_percent_best, '-.', dash_capstyle='round',  lw=2.0, label=u"Истинные модели")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,104))
#plt.legend(loc='lower left')
plt.figlegend((line1[0], line2[0], line3[0], line4[0]), (u"Маргинализация",u"Склеивание",u"Витерби", u"Истинные модели"),
              loc = 'center right')
#plt.tight_layout(pad=0.0,h_pad=0.01)
plt.show()

plt.savefig(filename+".png")

to_file = np.asarray([xs,ps1,ps_glue1,ps_viterbi1,
                      a_norms1,a_norms_glue1,a_norms_viterbi1,
                      b_norms1,b_norms_glue1,b_norms_viterbi1,
                      class_percent, class_percent_glue, class_percent_viterbi])
np.savetxt(filename+".csv", to_file.T, delimiter=';')

print("--- %.1f minutes ---" % ((time.time()-start_time) / 60))

#xs,ps1,ps_glue1,ps_viterbi1, a_norms1,a_norms_glue1,a_norms_viterbi1,b_norms1,\
#b_norms_glue1,b_norms_viterbi1, class_percent, class_percent_glue, class_percent_viterbi\
# = np.loadtxt(filename+".csv", delimiter=';', unpack=True)