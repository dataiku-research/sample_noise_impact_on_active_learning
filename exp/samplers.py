from cardinal.clustering import KMeansSampler, KCentroidSampler, MiniBatchKMeansSampler
from cardinal.zhdanov2019 import TwoStepKMeansSampler
import numpy as np
from cardinal.base import ScoredQuerySampler, BaseQuerySampler
from sklearn.neighbors import KNeighborsClassifier
from cardinal.metrics import exploration_score
from cardinal.uncertainty import MarginSampler, margin_score, EntropySampler, ConfidenceSampler, _get_probability_classes
from met import IncrementalMiniBatchKMeansSampler
import copy
from torch import tensor


def to_classes(data):
    if len(data.shape) == 2:
        return np.argmax(data, axis=1)
    return data


class TwoStepIncrementalMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       fixed_cluster_centers=None) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected], fixed_cluster_centers=fixed_cluster_centers)
        selected = selected[new_selected]
        
        return selected


class TwoStepMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):

        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            MiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       ) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected])
        selected = selected[new_selected]
        
        return selected


class InformedConfidenceSampler(ConfidenceSampler):
    def score_samples(self, X: np.array, max_confidence:np.array) -> np.array:
        score = super(InformedConfidenceSampler, self).score_samples(X)
        return np.min([score, max_confidence], axis=0)

    def select_samples(self, X: np.array, max_confidence:np.array) -> np.array:
        sample_scores = self.score_samples(X, max_confidence)
        self.sample_scores_ = sample_scores
        index = np.argsort(sample_scores)[-self.batch_size:]
        return index

class InformedMarginSampler(MarginSampler):
    def score_samples(self, X: np.array, max_margin:np.array) -> np.array:
        score = super(InformedMarginSampler, self).score_samples(X)
        return np.min([score, max_margin], axis=0)

    def select_samples(self, X: np.array, max_margin:np.array) -> np.array:
        sample_scores = self.score_samples(X, max_margin)
        self.sample_scores_ = sample_scores
        index = np.argsort(sample_scores)[-self.batch_size:]
        return index

class InformedEntropySampler(EntropySampler):
    def score_samples(self, X: np.array, max_entropy:np.array) -> np.array:
        score = super(InformedEntropySampler, self).score_samples(X)
        return np.min([score, max_entropy], axis=0)

    def select_samples(self, X: np.array, max_entropy:np.array) -> np.array:
        sample_scores = self.score_samples(X, max_entropy)
        self.sample_scores_ = sample_scores
        index = np.argsort(sample_scores)[-self.batch_size:]
        return index
