import sys
import json
from importlib import import_module
import dataset
import copy
import numpy as np
import pickle
import os
import shutil
import itertools
import time
from pathlib import Path
from collections import defaultdict


class Experiment():

    def __init__(self, db, seed, path='./cache', force=False, verbose=0):
        self.db = db
        self.seed = seed
        self.path = Path(path) / str(seed)
        if not self.path.exists():
            self.path.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self._db_conn = dataset.connect(self.db)
        self._memory = defaultdict(dict)
        self.force = force
    
    def _log(self, verbosity, message):
        if self.verbose >= verbosity:
            print(message)

    def log_value(self, config, key, value, **kwargs):
        table = self._db_conn[key]
        table.upsert(dict(value=value, **config, **kwargs), list(config.keys()))

    def _load(self, iter_id, name, tmp=False):
        if tmp:
            filebase = self.path / str(iter_id) / 'tmp' / name
        else:
            filebase = self.path / str(iter_id) / name 
        if filebase.with_suffix('.npy').exists():
            value = np.load(filebase.with_suffix('.npy'))
        elif filebase.with_suffix('.krs').exists():
            from keras.models import load_model
            value = load_model(filebase.with_suffix('.krs'))
        elif filebase.with_suffix('.pkl').exists():
            with open(filebase.with_suffix('.pkl'), 'rb') as filedesc:
                value = pickle.load(filedesc)
        else:
            raise ValueError('Could not load variable {}.{}'.format(iter_id, name))
        self._memory[iter_id][name] = value
        return value

    def _save_value_at(self, iter_id, name, value, tmp=False):
        self._memory[iter_id][name] = value

        if tmp:
            filebase = self.path / str(iter_id) / 'tmp' / name
        else:
            filebase = self.path / str(iter_id) / name
        if type(value).__module__ == np.__name__:
            np.save(filebase.with_suffix('.npy'), value)
        elif 'keras' in value.__module__.split('.'):
            from keras.models import save_model
            save_model(value, filebase.with_suffix('.krs'))
        else:
            with open(filebase.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(value, f)

    def retrieve_value_at(self, iter_id, name, first=None):
        self._log(2, 'Retrieving {} {}'.format(iter_id, name))
        if self.first:
            return first
        if iter_id in self._memory and name in self._memory[iter_id]:
            return self._memory[iter_id][name]
        return self._load(iter_id, name)

    def persist_value_at(self, iter_id, name, value):
        self._log(2, 'Persisting {} {}'.format(iter_id, name))
        self._memory[iter_id][name] = value
        self._save_value_at(iter_id, name, value)

    def resume_value_at(self, iter_id, name, first=None):
        self._log(2, 'Resuming {} {}'.format(iter_id, name))
        if self.first:
            return first
        if iter_id in self._memory and name in self._memory[iter_id]:
            return self._memory[iter_id][name]
        return self._load(iter_id, name, tmp=True)

    def cache_value_at(self, iter_id, name, value):
        self._log(2, 'Caching {} {}'.format(iter_id, name))

        self._memory[iter_id][name] = value
        self._save_value_at(iter_id, name, value, tmp=True)

    def iter(self, items, force_recompute=False):

        previous_iter_id = None
        self.first = True
        self._memory = defaultdict(dict)

        for current_iter_id in items:
            tmp_path = self.path / str(current_iter_id) / 'tmp'
            tmp_path.mkdir(exist_ok=True, parents=True)
            summary_path = self.path / str(current_iter_id) / 'completed.json'

            if summary_path.exists() and not self.force:
                self._log(1, 'Iteration {} already computed.'.format(current_iter_id))
                self.first = False
                continue

            t0 = time.time()
            yield current_iter_id
            delta = time.time() - t0

            with open(str(summary_path), 'w') as f:
                json.dump(dict(duration=delta), f)
            
            if previous_iter_id is not None:
                del self._memory[previous_iter_id]
            tmp_path = self.path / str(previous_iter_id) / 'tmp'
            if tmp_path.exists():
                shutil.rmtree(str(tmp_path))
            
            previous_iter_id = current_iter_id
            self.first = False

        tmp_path = self.path / str(previous_iter_id) / 'tmp'
        if tmp_path.exists():
            shutil.rmtree(str(tmp_path))
        del self._memory
        self._memory = defaultdict(dict)
