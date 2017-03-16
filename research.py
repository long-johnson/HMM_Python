# -*- coding: utf-8 -*-

#
# Various useful procedures to conduct my research on HMMs and missing data
#

import numpy as np
import GaussianHMM as ghmm
import StandardImputationMethods as impute 
import copy


def make_missing_values(seqs_orig, to_dissapears, n_of_gaps):
    """ make some observations missing in original sequences
    
    Parameters
    ----------
    seqs_orig : list (K) of float 2darrays (TxZ) or int 1darrays (T)
        original sequences
    to_dissapears : list (K) of int 1darray (T)
        shuffled indexes of observations from original sequences
    n_of_gaps : {int, (int min, int max)}
        if int - then how many gaps should be made in each sequence
        if (int min, int max) - then number of gaps will vary across sequences in range [min, max)

    Returns
    -------
    seqs : list (K) of float 2darrays (TxZ) or int 1darrays (T)
        seqs
    avails : list (K) of bool 1darrays (T)
        avails
    """
    K = len(seqs_orig)
    avails = [np.full(len(seqs_orig[i]), True, dtype=np.bool)
              for i in range(K)]
    seqs = copy.deepcopy(seqs_orig)
    for k in range(K):
        if type(n_of_gaps) is int:
            n_of_gaps_ = n_of_gaps
        else:
            n_of_gaps_ = np.random.randint(*n_of_gaps)
        avails[k][to_dissapears[k][:n_of_gaps_]] = False
        seqs[k][to_dissapears[k][:n_of_gaps_]] = np.nan
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


def make_svm_training_seqs(seqs_orig, to_dissapears, range_of_gaps):
    train_seqs = []
    train_avails = []
    for n_of_gaps in range_of_gaps:
        seqs, avails = make_missing_values(
                           seqs_orig, to_dissapears,
                           n_of_gaps
                       )
        train_seqs += seqs
        train_avails += avails
    return train_seqs, train_avails  


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
    if verbose == True:
        print("algorithm_class = {}, algorithm_gaps = {}"\
              .format(algorithm_class, algorithm_gaps))
    K = len(seqs1)
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
    if verbose == True:
        print("Correctly classified {} %".format(percent))


# TODO: make full support of DHMM evaluation !!!
def evaluate_decoding_imputation(decoded_percents, imputed_sum_squares_diff, step,
                                 hmm, seqs, states_list_orig,
                                 seqs_orig, avails, algorithm_gaps='viterbi',
                                 n_neighbours=10, verbose=False):
    """ Code to evaluate hmm decoding  and imputation performance on
    given sequences. n_neighbours works only with ‘mean’ or 'mode' algorithm
    
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
    if algorithm_gaps == 'mean' or algorithm_gaps == 'mode':
        imp_method = 'mean' if type(hmm) is ghmm.GHMM else 'mode'
        # impute
        seqs_imp, avails_imp = impute.impute_by_n_neighbours(
                                   seqs, avails, n_neighbours, method=imp_method
                               )
        seqs_imp = impute.impute_by_whole_seq(seqs_imp, avails_imp, method=imp_method)
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