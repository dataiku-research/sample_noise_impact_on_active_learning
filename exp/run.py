import os
import sys
import itertools
from copy import deepcopy
from pathlib import Path
import importlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

from cardinal.uncertainty import MarginSampler, ConfidenceSampler,  _get_probability_classes
from cardinal.random import RandomSampler
from cardinal.clustering import MiniBatchKMeansSampler
from cardinal.utils import ActiveLearningSplitter

from samplers import TwoStepIncrementalMiniBatchKMeansSampler, TwoStepMiniBatchKMeansSampler, InformedConfidenceSampler
from experimenter import Experiment


samplers_to_compute = None
if len(sys.argv) > 1:
    samplers_to_compute = sys.argv[1:]
    print('Will only compute samplers', samplers_to_compute)


cwd = Path.cwd()
cache_path = cwd / 'cache'
exp_path = cwd / '..' / 'exp'
database_path = 'sqlite:///database.db'

dataset_name = cwd.stem


print('Cache path is {}'.format(cache_path))

# Load experiment configuration
exp_module = importlib.import_module(dataset_name)
exp_config = exp_module.get_config()

start_size = exp_config['start_size']
batches = exp_config['batches']
oracle_error = exp_config.get('oracle_error', None)

data = exp_module.get_dataset()
X = data['X']
y = data['y']

get_clf = exp_module.get_clf
fit_clf = exp_module.fit_clf

y_ = y
if len(y.shape) == 2:
    y_ = np.argmax(y, axis=1)


if len(y.shape) == 2:
    n_classes = y.shape[1]
else:
    n_classes = len(np.unique(y))


k_start = False

iters = [i.item() for i in np.cumsum([start_size] + batches)]
batches.append(batches[-1])
assert(len(batches) == len(iters))


model_cache = dict()


for seed, ds in itertools.product(['11', '22', '33', '44', '55'], ['A', 'B']):
    print(seed, ds)
    methods = {
        'random': lambda params: RandomSampler(batch_size=params['batch_size'], random_state=int(seed)),
        'margin': lambda params: MarginSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'uncertainty': lambda params: ConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
        'wkmeans': lambda params: TwoStepMiniBatchKMeansSampler(n_classes, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
        'iwkmeans': lambda params: TwoStepIncrementalMiniBatchKMeansSampler(n_classes, params['clf'], params['batch_size'], assume_fitted=True, n_init=1, random_state=int(seed)),
        'iconfidence': lambda params: InformedConfidenceSampler(params['clf'], batch_size=params['batch_size'], assume_fitted=True),
    }
    if samplers_to_compute is None:
        samplers_to_compute = list(methods.keys())
    
    splitter = ActiveLearningSplitter(X.shape[0], test_size=.5, random_state=int(seed))
    index = np.arange(X.shape[0])

    precomputed_proba_path = Path('precomputed_proba') / (seed + ds)

    if not precomputed_proba_path.exists():
        precomputed_proba_path.mkdir(parents=True)
        clf = get_clf()
        fit_clf(clf, X[splitter.test], y[splitter.test], **exp_config.get('full_dataset_fit_params', {}))
        y_proba = _get_probability_classes(clf, X)

        max_confidence = confidence_score('precomputed', y_proba)
        np.save(str(precomputed_proba_path / 'max_confidence.npy'), max_confidence)
        np.save(str(precomputed_proba_path / 'proba.npy'), 1 - max_confidence)

    max_confidence = np.load(str(precomputed_proba_path / 'max_confidence.npy'))

    for name in samplers_to_compute:
        print(name)
        
        splitter = ActiveLearningSplitter(X.shape[0], test_size=.5, random_state=int(seed))
        if ds == 'B':
            # Exchange -1 and -2
            splitter._mask = -(splitter._mask + 2) - 1

        method = methods[name]
        exp = Experiment(database_path, seed + ds, path=os.path.join(cache_path, name))

        if not k_start:
            first_index, _ = train_test_split(np.arange(X[splitter.train].shape[0]), train_size=iters[0], random_state=int(seed), stratify=y[splitter.train])
        else:
            start_sampler = MiniBatchKMeansSampler(iters[0], random_state=int(seed))
            start_sampler.fit(X[splitter.train])
            first_index = start_sampler.select_samples(X[splitter.train])
        splitter.add_batch(first_index)
        
        for i in exp.iter(range(len(iters))):
            selected = exp.retrieve_value_at(i - 1, 'selected', first=splitter._mask)
            splitter._mask = selected

            classifier = exp.resume_value_at(i - 1, 'classifier', first=get_clf())
            fit_clf(classifier, X[splitter.selected], y[splitter.selected])
            exp.cache_value_at(i, 'classifier', classifier)

            predicted = exp.persist_value_at(i, 'predicted', classifier.predict_proba(X))
     
            params = dict(batch_size=batches[i], clf=classifier, iter=i + 1)
            sampler = method(params)
            sampler.fit(X[splitter.selected], y[splitter.selected])

            if name.startswith('iwkmeans'):
                new_selected_index = sampler.select_samples(X[splitter.non_selected], fixed_cluster_centers=X[splitter.selected])
            elif name.startswith('iconfidence'):
                new_selected_index = sampler.select_samples(X[splitter.non_selected], max_confidence[splitter.non_selected])
            else:
                new_selected_index = sampler.select_samples(X[splitter.non_selected])

            splitter.add_batch(new_selected_index)
            exp.persist_value_at(i, 'selected', splitter._mask)
            print(iters[i], splitter.selected.sum())

            
        for i in range(len(iters)):
            selected = exp.retrieve_value_at(i, 'selected')
            splitter._mask = selected
            splitter.current_iter = selected.max()
            assert(splitter.selected_at(i).sum() == iters[i])
            predicted = exp.retrieve_value_at(i, 'predicted')
            predicted_test = predicted[splitter.test] 
            predicted_selected = predicted[splitter.selected_at(i)] 
            
            config = dict(
                seed=seed + ds,
                method=name,
                n_iter=i,
                dataset=dataset_name
            )

            exp.log_value(config, 'accuracy', accuracy_score(y_[splitter.test], np.argmax(predicted_test, axis=1)))
            exp.log_value(config, 'selected_accuracy', accuracy_score(y_[splitter.selected_at(i)], np.argmax(predicted_selected, axis=1)))
            print('acc', accuracy_score(y_[splitter.test], np.argmax(predicted_test, axis=1)))
