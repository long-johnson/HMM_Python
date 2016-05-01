# -*- coding: utf-8 -*-

import copy
import numpy as np

# TODO: more universal mode
def impute_by_n_neighbours(seqs, avails, n, is_middle=True, method='avg',
                           n_of_symbols=2):
    """ Impute missing values (imp) by values of its neighbours
    method -- "average": imp = avg of n neighbours
              "mode":    imp = mode of n neighbours
    n_of_symbols -- int k: number of distinct values (to be used with "mode" method)
    is_middle -- true: take ceil(n/2) next and ceil(n/2) prev values
                 false: take n previous values
    """
    assert method in ('avg', 'mode'), "Invalid method '{}'".format(mode)
    K = len(seqs)
    seqs_imp = []
    avails_imp = []
    for k in range(K):
        seq = seqs[k]
        avail = avails[k]
        seq_imp, avail_imp = _impute_by_n_neighbours(seq, avail, n, is_middle,
                                                     method, n_of_symbols)
        seqs_imp.append(seq_imp)
        avails_imp.append(avail_imp)
    return seqs_imp, avails_imp
        
def _impute_by_n_neighbours(seq, avail, n_, is_middle, method, n_of_symbols):
    T = seq.size
    seq_imp = np.array(seq)
    avail_imp = np.array(avail)
    if is_middle:
        n = np.int32(np.ceil(n_/2.0))
    else:
        n = n_
    # TODO: optimize
    for t in range(T):
        if not avail[t]:
            l_b = t-n
            if l_b < 0: l_b = None
            if is_middle:
                r_b = t+n+1
                if T <= r_b: r_b = None
            else:
                r_b = t
            avl = avail[l_b:r_b]
            n_of_avl = np.count_nonzero(avl)
            # if no neigbours availiable
            if n_of_avl == 0:
                continue
            if method == 'avg':
                imp = 1.0 * np.sum((seq_imp[l_b:r_b])[avl]) / n_of_avl
            if method == 'mode':
                imp = np.argmax(np.histogram((seq_imp[l_b:r_b])[avl],\
                      bins=np.arange(n_of_symbols+1))[0])
            seq_imp[t] = imp
            avail_imp[t] = True
    return seq_imp, avail_imp
    
# TODO: more universal mode
# TODO: what if no observations are availiable at all?
def impute_by_whole_seq(seqs, avails, method="avg", n_of_symbols=2):
    """ Impute missing values (imp) with average value
    method -- "average": imp = avg of sequence
    "mode":    imp = mode of sequence
    """
    K = len(seqs)
    seqs_imp = []
    for k in range(K):
        seq_imp = np.array(seqs[k])
        avail = avails[k]
        if method == 'avg':
            imp = np.sum(seq_imp[avail]) / np.count_nonzero(avail)
        else:
            imp = np.argmax(np.histogram(seq_imp[avail],\
                      bins=np.arange(n_of_symbols+1))[0])
        # set all missing values to the imputed value
        seq_imp[np.logical_not(avail)] = imp
        seqs_imp.append(seq_imp)
    return seqs_imp

#seqs = [np.array([1,0,0,0,0])]
#avails = [np.array([True,False,True,False,False], dtype=np.bool)]
#seqs[0][np.logical_not(avails[0])] = -1
#print impute_by_n_neighbours(seqs, avails, n=1, is_middle=True, method="mode",
#                            n_of_symbols=3)

#print impute_by_whole_seq(seqs, avails, method="mode", n_of_symbols=3)