import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from matplotlib import cm
import matplotlib
from cardinal.clustering import KMeansSampler, KCentroidSampler, MiniBatchKMeansSampler
from cardinal.zhdanov2019 import TwoStepKMeansSampler


names = {

    # Datasets
    # --------

    "noisy": "Noisy",
    "noisy_hd": "Noisy HD",
    "noisy_ld": "Noisy LD",
    "cifar10": "CIFAR-10",
    "cifar10_v3": "CIFAR-10",
    "ldpa_noisy": "LDPA noisy",
    "covertype": "Covertype",
    "cifar10_simclr": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "fashion": "Fashion",
    "nomao": "NOMAO",
    "phishing": "Phishing",
    "news": "20 NG",
    "ldpa": "LDPA",
    "wall_robot": "Robot",

    # Classics
    # --------

    "uncertainty": "Confidence",
    "confidence": "Confidence",
    "margin": "Margin",
    "entropy": "Entropy",

    # Informed
    # --------

    "iconfidence": "IConfidence",
    "imargin": "IMargin",
    "ientropy": "IEntropy",

    # Pure explorer
    # -------------

    "kmeans": "KMeans",
    "random": "Random",
    "wrandom": "WRandom",
    "ikmeans": "IKMeans",

    # Zhdanov
    # -------

    "wkmeans": "WKMeans",
    "wkmeans50": "WKMeans 50",
    "iwkmeans": "IWKMeans",
    "iwkmeans5": "IWKMeans 5",
    "iwkmeans10": "IWKMeans 10",
    "iwkmeans20": "IWKMeans 20",
    "iwkmeans40": "IWKMeans 40",
    "iwkmeans80": "IWKMeans 80",
    "wkmeans80": "WKMeans 80",
    "iwkmeans2": "IWKMeans Fixed",

    # Others
    # ------

    "wkdiff": "WKDiff-L2",
    "wkdiffl1": "WKDiff-L1",
    "bbald": "BatchBALD",
    "bald": "BALD",
    "wbald": "WBALD",
    "wbbald": "WBatchBALD",
    "eemargin": "EEMargin",
    "adaptive": "Adaptive",
    "adaptive2": "Adaptive v2",
    "eeconfidence": "EEConfidence",
    "accuracy": "Accuracy",
    "hard_contradiction": "Contradiction ratio",
    "top_exploration": "Exploration",
    "batch_difficulty": "Batch easiness ratio",
    "batch_agreement": "Batch classifier agreement ratio",
    "iconfidence": "Informed confidence (oracle)",
}

def namify(tag):
    return tag.replace('_', ' ').capitalize()

# We want to have a uniform selection of style / colors in all plots
mpl_options = {
    'random': dict(c=cm.tab10(0), linestyle='solid'),

    'entropy': dict(c=cm.tab10(1), linestyle='dashed'),
    'uncertainty': dict(c=cm.tab10(5), linestyle='dashed'),
    'margin': dict(c=cm.tab10(4), linestyle='dashed'),

    'ientropy': dict(c=cm.tab10(1), linestyle=(0, (3, 1, 1, 1))),
    'iconfidence': dict(c=cm.tab10(5), linestyle=(0, (3, 1, 1, 1))),
    'imargin': dict(c=cm.tab10(4), linestyle=(0, (3, 1, 1, 1))),


    'wkmeans': dict(c=cm.tab10(2), linestyle='dashed'),
    'wkmeans50': dict(c=cm.tab10(2), linestyle='dotted'),
    'kmeans': dict(c=cm.tab10(3), linestyle='dashdot'),
    'iwkmeans': dict(c=cm.tab10(6), linestyle='dashed'),

    'ikmeans': dict(c=cm.tab10(7), linestyle='dashed'),
    'wrandom': dict(c=cm.tab10(8), linestyle='dashed'),
    'wkdiff': dict(c=cm.tab10(9), linestyle='dashdot'),
    'wkdiffl1': dict(c=cm.tab20(2), linestyle='dashdot'),
    'bbald': dict(c=cm.tab20(4), linestyle='dashdot'),
    'bald': dict(c=cm.tab20(6), linestyle='dotted'),
    'wbald': dict(c=cm.tab20(8), linestyle='dotted'),
    'wbbald': dict(c=cm.tab20(10), linestyle='dotted'),
    'eemargin': dict(c=cm.tab20(12), linestyle='dotted'),
    'eeconfidence': dict(c=cm.tab20(14), linestyle='dotted'),
    'adaptive': dict(c=cm.tab10(7), linestyle='dotted'),
    'adaptive2': dict(c=cm.tab20(18), linestyle='dotted'),
    'iwkmeans2': dict(c=cm.tab20(19), linestyle='dotted'),
    'iwkmeans3': dict(c=cm.tab20(18), linestyle='dotted'),
    'iwkmeans4': dict(c=cm.tab10(3), linestyle='dotted'),
    "iwkmeans5": dict(c=cm.tab20(1)),
    "iwkmeans10": dict(c=cm.tab20(3)),
    "iwkmeans20": dict(c=cm.tab20(5)),
    "iwkmeans40": dict(c=cm.tab20(7)),
    "iwkmeans80": dict(c=cm.tab10(8)),
    "idwkmeans40": dict(c=cm.tab20(7), linestyle='dotted'),
    "idwkmeans20": dict(c=cm.tab20(11), linestyle='dotted'),
    "idwkmeans": dict(c=cm.tab20(9), linestyle='dotted'),
    "wkmeans10": dict(c=cm.tab20(13)),
    "wkmeans20": dict(c=cm.tab20(15)),
    "wkmeans40": dict(c=cm.tab20(17)),
    "wkmeans80": dict(c=cm.tab10(9)),
    "iwkmeansa": dict(c=cm.tab20(8)),
    "adaptive3": dict(c=cm.tab10(1)),
    "adaptive4": dict(c=cm.tab10(1)),
    "cifar10": dict(c=cm.tab20(1)),
    "cifar100": dict(c=cm.tab20(3)),
    "mnist": dict(c=cm.tab20(5)),
    "fashion": dict(c=cm.tab20(7)),
    "nomao": dict(c=cm.tab20(9)),
    "phishing": dict(c=cm.tab20(11)),
    "news": dict(c=cm.tab20(13)),
    "ldpa": dict(c=cm.tab20(15)),
    "wall_robot": dict(c=cm.tab20(17)),
}


def init_figure():
    plt.figure(figsize=(10, 8))
    plt.grid(zorder=0)


def plot_by_method(df, cumsum=False, log=False):

    transform = lambda x: x
    if cumsum:
        transform = lambda x: np.cumsum(x)
    is_iter = False

    # df = df[df['seed'] =='11A']
    for method, mdf in df.groupby('method'):
        #mdf = mdf.drop_duplicates(subset=['dataset', 'method', 'seed', 'n_samples'], keep='first')

        #print(method)
        #print(mdf)

        gmdf = mdf.groupby('n_samples').agg([
            ('mean',lambda x: np.mean(transform(x))),
            ('q10', lambda x: np.quantile(transform(x), 0.1, axis=0)),
            ('q90', lambda x: np.quantile(transform(x), 0.9, axis=0))
        ])['value'].sort_index()
    
        x = gmdf.index.values
        mean = gmdf['mean'].values
        q10 = gmdf['q10'].values
        q90 = gmdf['q90'].values
    
        # Plot the mean line and get its color
        line = plt.plot(x, mean, label=names.get(method, namify(method)), **mpl_options[method])
        color = line[0].get_c()
    
        # Plot confidence intervals
        plt.fill_between(x, q90, q10, alpha=.3, color=color, zorder=2)
        
        if log:
            plt.xscale('log')
            plt.minorticks_off()
            scale_loc = (np.arange(x.size) ** 1.5).astype(int)
            scale_loc = scale_loc[scale_loc < x.size]
            plt.gcf().autofmt_xdate()
            plt.gca().set_xticks(x[scale_loc])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return is_iter

def save_figure(dataset, suffix, metric, is_iter, legend_kwargs={}):

    if legend_kwargs is not None:
        plt.legend(**legend_kwargs)
    if is_iter:
        plt.xlabel('Iteration')
    else:
        plt.xlabel('Training sample count')
    plt.ylabel(names.get(metric, namify(metric)))
    
    plt.savefig('{}{}_{}.pdf'.format(dataset, suffix, metric), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_boxplot(df, cumsum=False):

    transform = lambda x: x
    if cumsum:
        transform = lambda x: np.cumsum(x)
    is_iter = False
    if 'n_iter' in df.columns:
        df['n_samples'] = df['n_iter']
        is_iter = True

    gmdf = df[df['n_samples'] == df['n_samples'].max()][['method', 'value']].sort_index()
    gmdf.boxplot(by='method')
    return is_iter


def plot_table(df, cumsum=False, last=False, normalize=False):

    transform = lambda x: x
    if cumsum:
        transform = lambda x: np.cumsum(x)

    max_samples = df['n_samples'].max()

    if last:
        gmdf = df[df['n_samples'] == max_samples]
    if cumsum:
        gmdf = df.groupby(['method', 'seed']).mean().reset_index()
    if normalize:
        gmdf['value'] = gmdf['value'] / max_samples

    gmdf = gmdf[['method', 'value']]

    gmdf = gmdf.sort_index()
    print(gmdf.groupby('method').describe().T.to_latex(float_format="%.3f"))
