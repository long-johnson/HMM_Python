# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import itertools
import GaussianHMM as ghmm
import warnings
import sklearn
from sklearn.model_selection import train_test_split
import copy
import StandardImputationMethods as imp
warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO: one vs all classification
#
# Parameters
#
N = 3
M = 3
Z = 3
len_seq = 100
rtol = 1e-5
max_iter = 1000
hmms0_size = 2
n_classes = 23
test_size = 0.25
folder = "data/User Identification From Walking Activity Data Set/"

#
# Functions
#
def scale_features(map_class_seqs, inplace=True, scale_params=None):
    max_feats = []
    min_feats = []
    for cl in map_class_seqs:
        max_feats.append(np.max(map_class_seqs[cl], axis=0))
        min_feats.append(np.min(map_class_seqs[cl], axis=0))
    max_feats = np.max(max_feats, axis=0)
    min_feats = np.min(min_feats, axis=0)
    for cl in map_class_seqs:
        map_class_seqs[cl] = (map_class_seqs[cl] - min_feats) / (max_feats - min_feats)
    return {'max_feats': max_feats, 'min_feats': min_feats}


#==============================================================================
# load
#==============================================================================

def load_user_id_data(folder, n_classes, len_seq):
    map_class_seqs = dict()
    for i in range(1, min(n_classes, 22+1)):
        import os.path
        fname = folder + "{}.csv".format(i)
        if not os.path.isfile(fname):
            continue
        data = pd.read_csv(fname, header=None, index_col=0,
                           names=['time', 'x', 'y', 'z'])
        n_seq = len(data) // len_seq
        data = data.iloc[0:(n_seq * len_seq), :]
        map_class_seqs[str(i)] = np.split(data.values, n_seq, axis=0)
    return map_class_seqs

print('loading data')
map_class_seqs = load_user_id_data(folder, n_classes, len_seq)
print('data loaded')
print()

#===============================================================================
# Train-test split
#===============================================================================
map_class_seqs_train = dict()
map_class_seqs_test = dict()
for person in map_class_seqs.keys():
    seqs = map_class_seqs[person]
    train, test = train_test_split(seqs, test_size=test_size, random_state=42)
    map_class_seqs_train[person] = train
    map_class_seqs_test[person] = test

#==============================================================================
# proper scaling
#==============================================================================
#scale_params_train = map_class_seqs_train(map_class_seqs, Z, inplace=True)
#scale_params = scale_features(map_class_seqs, inplace=True)

#==============================================================================
# Train corresponding HMMS
#==============================================================================
np.random.seed(42)
start_time = time.time()
hmms = []
for activity, seqs in zip(map_class_seqs_train.keys(), map_class_seqs_train.values()):
    print('training ''{}'' hmm'.format(activity))
    hmm, p_max, iter_best, n_of_best = \
        ghmm.train_best_hmm_baumwelch(seqs, hmms0_size=hmms0_size, N=N, M=M, Z=Z,
                                      rtol=rtol, max_iter=max_iter,
                                      covariance_type='diag',
                                      verbose=False)
    hmms.append(hmm)
#    print('pi')
#    print(hmm._pi)
#    print('a')
#    print(hmm._a)
#    print('tau')
#    print(hmm._tau)
#    print('mu')
#    for i, j in itertools.product(range(N), range(M)):
#        print('mu[{}, {}]:\n{}'.format(i, j, hmm._mu[i, j]))
#    print('sig')
#    for i, j in itertools.product(range(N), range(M)):
#        print('sig[{}, {}]:\n{}'.format(i, j, hmm._sig[i, j]))
    print('info')
    print('p_max', p_max)
    print('iter_best', iter_best)
    print('n_of_best', n_of_best)
    print('time: {:.1f} m'.format( (time.time() - start_time) / 60))
    print('-------------------')
    print()
print()


#==============================================================================
# Classify train data
#==============================================================================
# print('classify train')
# accuracies = []
# for i, (activity, seqs) in enumerate(zip(map_class_seqs_train.keys(),
#                                          map_class_seqs_train.values())):
#     pred = ghmm.classify_seqs_mlc(seqs, hmms)
#     acc = 100.0 * np.count_nonzero(pred == i) / len(pred)
#     accuracies.append(acc)
#     print('Person ''{}'' acc: {:.1f}%'.format(activity, acc))
# print('train np.mean(accuracies)', np.mean(accuracies))
# print('time: {:.1f} m'.format( (time.time() - start_time) / 60))


#==============================================================================
# Classify test data
#==============================================================================
# print('classify test')
# accuracies = []
# for i, (activity, seqs) in enumerate(zip(map_class_seqs_test.keys(),
#                                          map_class_seqs_test.values())):
#     pred = ghmm.classify_seqs_mlc(seqs, hmms)
#     acc = 100.0 * np.count_nonzero(pred == i) / len(pred)
#     accuracies.append(acc)
#     print('Person ''{}'' acc: {:.1f}%'.format(activity, acc))
# print('test np.mean(accuracies)', np.mean(accuracies))
# print('time: {:.1f} m'.format( (time.time() - start_time) / 60))
# print('N, M, len_seq, rtol, max_iter, hmms0_size',
#       N, M, len_seq, rtol, max_iter, hmms0_size)


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

def make_missing_values(seqs_train_orig, to_dissapears, n_of_gaps):
    K = len(seqs_train_orig)
    avails = [np.full(len(seqs_train_orig[k]), True, dtype=np.bool)
              for k in range(K)]
    seqs_train = copy.deepcopy(seqs_train_orig)
    for k in range(K):
        avails[k][to_dissapears[k][:n_of_gaps]] = False
        seqs_train[k][to_dissapears[k][:n_of_gaps]] = np.nan
    return seqs_train, avails


#==============================================================================
# Classify/decode/imp on train / test data with missing values
#==============================================================================
np.random.seed(42)
GAP_PERCENTS = list(np.arange(0.0, 0.91, 0.1))
ALGORITHMS = ['viterbi', 'mean', 'marginalization', 'gluing', ]
IS_GAPS_PLACES_DIFFERENT = False
# TODO: cross-validation ?
print('Classification with missing data')
# сгенерируем случайные индексы пропусков в последовательностях
map_class_to_disappears = dict()
for person in map_class_seqs_test.keys():
    seqs = map_class_seqs_test[person]
    map_class_to_disappears[person] = \
        gen_gaps_positions(len(seqs), len_seq, is_gaps_places_different=True)


# результирующий массив с точностями по процентам пропусков
res_accs = dict()
res_mses = dict()
for algorithm in ALGORITHMS:
    print(algorithm)
    res_accs[algorithm] = []
    for gap_percent in GAP_PERCENTS:
        print(gap_percent)
        # наделаем пропусков в последовательностях
        map_class_seqs_test_gaps = dict()
        map_class_avails_test = dict()
        for person in map_class_seqs_test.keys():
            seqs = map_class_seqs_test[person]
            to_disappears = map_class_to_disappears[person]
            n_of_gaps = int(len_seq * gap_percent)
            seqs, avails = make_missing_values(seqs, to_disappears, n_of_gaps)
            map_class_seqs_test_gaps[person] = seqs
            map_class_avails_test[person] = avails
        # 1) классификация последовательностей с пропусками
        class_accuracies = []
        for i, person in enumerate(map_class_seqs_test_gaps.keys()):
            seqs = map_class_seqs_test_gaps[person]
            avails = map_class_avails_test[person]
            pred = ghmm.classify_seqs_mlc(seqs, hmms, avails, algorithm)
            acc = 100.0 * np.count_nonzero(pred == i) / len(pred)
            class_accuracies.append(acc)
        # добавим полученную среднюю точность в массив точностей
        res_accs[algorithm].append(np.mean(class_accuracies))

        # 2) восстановление пропусков в последовательностях с пропусками
        impute_errors = []
        for i, person in enumerate(map_class_seqs_test_gaps.keys()):
            hmm = hmms[i] # type: ghmm.GHMM
            seqs_orig = map_class_seqs_test[person]
            seqs = map_class_seqs_test_gaps[person]
            avails = map_class_avails_test[person]
            if algorithm == 'viterbi':
                # восстановление с помощью алгоритма Витерби
                states_list = hmm.decode_viterbi(seqs, avails)
                imputed = hmm.impute_by_states(seqs, avails, states_list)
            else:
                # восстановление с помощью ср. арифм. соседей
                seqs_imp, avails_imp = imp.impute_by_n_neighbours(seqs, avails, 10)
                imputed = imp.impute_by_whole_seq(seqs, avails)
            # среднекв. ошибка по каждой последовательности
            mses = [np.sum((seq - seq_orig) ** 2) / len(seq) \
                    for seq, seq_orig in zip(seqs, seqs_orig)]
            impute_errors += mses
        # усредним среднюю ошибку по всем последовательностям
        res_mses[algorithm].append(np.mean(impute_errors))

pd.DataFrame(res_accs).to_excel('out/user_id_test_accs.xlsx')
pd.DataFrame(res_mses).to_excel('out/user_id_test_mses.xlsx')

for algorithm in ALGORITHMS:
    plt.plot(GAP_PERCENTS, res_accs[algorithm], label=algorithm)
plt.legend()
plt.show()

for algorithm in ALGORITHMS:
    plt.plot(GAP_PERCENTS, res_mses[algorithm], label=algorithm)
plt.legend()
plt.show()


