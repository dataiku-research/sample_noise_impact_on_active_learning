import dataset
import pandas as pd
from sklearn.metrics import auc
from autorank._util import *
from scipy import stats
from autorank import autorank, plot_stats
from matplotlib import pyplot as plt
from study import names
from pathlib import Path


def my_autorank(data, alpha=0.05, verbose=False, order='descending', approach='frequentist', rope=0.1, rope_mode='effsize',
             nsamples=50000, effect_size=None):
    # Bonferoni correction for normality tests
    alpha_normality = alpha / len(data.columns)
    all_normal, pvals_shapiro = test_normality(data, alpha_normality, verbose)

    # homogeneity needs only to be checked for frequentist approach
    if all_normal:
        if verbose:
            print("Using Bartlett's test for homoscedacity of normally distributed data")
        homogeneity_test = 'bartlett'
        pval_homogeneity = stats.bartlett(*data.transpose().values).pvalue
    else:
        if verbose:
            print("Using Levene's test for homoscedacity of non-normal data.")
        homogeneity_test = 'levene'
        pval_homogeneity = stats.levene(*data.transpose().values).pvalue
    var_equal = pval_homogeneity >= alpha
    if verbose:
        if var_equal:
            print("Fail to reject null hypothesis that all variances are equal "
                  "(p=%f>=%f)" % (pval_homogeneity, alpha))
        else:
            print("Rejecting null hypothesis that all variances are equal (p=%f<%f)" % (pval_homogeneity, alpha))

    res = rank_multiple_nonparametric(data, alpha, verbose, all_normal, order, effect_size)
    # need to reorder pvals here (see issue #7)
    return RankResult(res.rankdf, res.pvalue, res.cd, res.omnibus, res.posthoc, all_normal, pvals_shapiro,
                      var_equal, pval_homogeneity, homogeneity_test, alpha, alpha_normality, len(data), None, None,
                      None, None, res.effect_size)



datasets = [
    'cifar10',
    'cifar10_simclr',
    'cifar100',
    'mnist',
    'fashion',
    'ldpa',
]

names['iconfidence'] = 'IConfidence'
names['kcenter'] = 'KCenter'


for ds in datasets:
    # Load database
    accuracy = pd.read_csv(str(Path(ds) / 'results' / 'accuracy.csv'))

    accuracy = accuracy[accuracy['method'].isin(['iwkmeans', 'wkmeans', 'kcenter', 'uncertainty', 'iconfidence', 'random'])]

    accuracy['method'] = accuracy['method'].replace(names)

    # compute auc
    accuracy = accuracy.drop_duplicates(subset=['dataset', 'method', 'seed', 'n_iter'])
    f = lambda x: auc(x['n_iter'], x['value'])
    accuracy = accuracy.groupby(['dataset', 'method', 'seed']).apply(f)

    df = accuracy.unstack(['method']).reset_index()
    df = df.drop(['dataset', 'seed'], axis=1)
    df.columns.name = None

    print(df)

    result = my_autorank(df, verbose=True)

    plot_stats(result)
    plt.savefig('method_ranking_{}.pdf'.format(ds))
