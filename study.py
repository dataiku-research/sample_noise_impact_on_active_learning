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

    "noisy_hd": "Noisy HD",
    "noisy_ld": "Noisy LD",
    "cifar10": "CIFAR-10",
    "cifar10_simclr": "CIFAR-10 SimCLR",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "fashion": "Fashion",
    "ldpa": "LDPA",

    # Classics
    # --------

    "uncertainty": "Confidence",
    "confidence": "Confidence",
    "margin": "Margin",
    "entropy": "Entropy",

    # Informed
    # --------

    "iconfidence": "Informed confidence (oracle)",

    # Pure explorer
    # -------------

    "random": "Random",
    "kcenter": "KCenterGreedy",

    # Zhdanov
    # -------

    "wkmeans": "WKMeans",
    "iwkmeans": "IWKMeans",

    'noisy_in_selected': 'Noisy samples in training dataset'
}

def namify(tag):
    return tag.replace('_', ' ').capitalize()

# We want to have a uniform selection of style / colors in all plots
mpl_options = {
    'random': dict(c='gray', linestyle='solid'),

    'entropy': dict(c=cm.tab10(1), linestyle='dashed'),
    'uncertainty': dict(c=cm.tab10(4), linestyle='dashed'),
    'margin': dict(c=cm.tab10(5), linestyle='dashed'),

    'iconfidence': dict(c=cm.tab10(0), linestyle=(0, (3, 1, 1, 1))),

    'wkmeans': dict(c=cm.tab10(2), linestyle='dashed'),
    'iwkmeans': dict(c=cm.tab10(3), linestyle='dashed'),
    'kcenter': dict(c='orangered', linestyle='solid'),

    "cifar10": dict(c=cm.tab20(1)),
    "cifar100": dict(c=cm.tab20(3)),
    "mnist": dict(c=cm.tab20(5)),
    "fashion": dict(c=cm.tab20(7)),
    "ldpa": dict(c=cm.tab20(15)),
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
