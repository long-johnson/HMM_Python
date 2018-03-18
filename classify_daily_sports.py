# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from itertools import product as it_product
import GaussianHMM as ghmm
import warnings
import time
import itertools
import matplotlib.pylab as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_daily_sports():
    root_path = 'data\Daily and Sports activity recognition'

    path = Path(root_path)
    data = []
    for activity_id, activity_dir in enumerate(path.iterdir()):
        if not activity_dir.is_dir():
            continue
        persons = []
        for person_id, person_dir in enumerate(activity_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            seqs = []
            for segment_id, segment_file in enumerate(person_dir.iterdir()):
                if segment_file.is_dir():
                    continue
                seq = []
                with segment_file.open() as file:
                    for line in file.readlines():
                        features = list(map(float, line.split(',')))[0:1]
                        seq.append(features)
                seqs.append(seq)
            persons.append(seqs)
        data.append(persons)

#    n_activities = len(data)
    n_persons = len(data[0])
    n_sequences = len(data[0][0])
#    n_observations = len(data[0][0][0])
    n_dimensions = len(data[0][0][0][0]) if type(data) is list else 1


    with path.joinpath('activity_names.txt').open() as file:
        activity_names = list(map(lambda x: x.strip(), file.readlines()))

    map_class_seqs = dict()
    for i, activity in enumerate(activity_names):
        seqs = []
        for person_id, sequence_id in it_product(range(n_persons), range(n_sequences)):
            seqs.append(np.array(data[i][person_id][sequence_id]))
        map_class_seqs[activity] = seqs

    return map_class_seqs, n_dimensions


def scale_features(map_class_seqs, Z, inplace=True, scale_params=None):
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
print('loading data')
map_class_seqs, Z = load_daily_sports()
print('data loaded')
print()


#==============================================================================
# Histogram
#==============================================================================
for activity, seqs in zip(map_class_seqs.keys(), map_class_seqs.values()):
    sample = np.array(seqs[0]).flatten()
    plt.figure()
    plt.title(activity)
    plt.hist(sample)
    plt.savefig("out/daily_activities_histograms/{}.png" \
                .format(activity.replace(' ', '_').replace('/', '')))
    plt.close()
    #plt.show()
raise('asd')

#==============================================================================
# preprocessing
#==============================================================================
scale_params = scale_features(map_class_seqs, Z, inplace=True)


#==============================================================================
# Train corresponding HMMS
#==============================================================================
np.random.seed(42)
N = 2
M = 1
start_time = time.time()
hmms = []
for activity, seqs in zip(map_class_seqs.keys(), map_class_seqs.values()):
    print('training ''{}'' hmm'.format(activity))
    hmm, p_max, iter_best, n_of_best = \
        ghmm.train_best_hmm_baumwelch(seqs, hmms0_size=1, N=N, M=M, Z=Z,
                                      rtol=1e-10, max_iter=100,
                                      covariance_type='diag',
                                      verbose=False)
    hmms.append(hmm)
    print('pi')
    print(hmm._pi)
    print('a')
    print(hmm._a)
    print('tau')
    print(hmm._tau)
    print('mu')
    for i, j in itertools.product(range(N), range(M)):
        print('mu[{}, {}]:\n{}'.format(i, j, hmm._mu[i, j]))
    print('sig')
    for i, j in itertools.product(range(N), range(M)):
        print('sig[{}, {}]:\n{}'.format(i, j, hmm._sig[i, j]))
    print('info')
    print('p_max', p_max)
    print('iter_best', iter_best)
    print('n_of_best', n_of_best)
    print('time: {:.1f} m'.format( (time.time() - start_time) / 60))
    print('-------------------')
    print()
    break
print()


#==============================================================================
# Classify train data
#==============================================================================
accuracies = []
for i, (activity, seqs) in enumerate(zip(map_class_seqs.keys(), map_class_seqs.values())):
    pred = ghmm.classify_seqs_mlc(seqs, hmms)
    acc = 100.0 * np.count_nonzero(pred == i) / len(pred)
    accuracies.append(acc)
    print('activity ''{}'' acc: {:.1f}%'.format(activity, acc))
    #print('time: {:.1f} s'.format( (time.time() - start_time) / 60))
    break

print('np.mean(accuracies)', np.mean(accuracies))

print('time: {:.1f} m'.format( (time.time() - start_time) / 60))