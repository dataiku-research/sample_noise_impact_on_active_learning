import dataset
import pandas as pd
from sklearn.metrics import auc
from autorank import autorank, plot_stats
from matplotlib import pyplot as plt
from study import names
from pathlib import Path


datasets = [
    'cifar10',
    'cifar10_simclr',
    'cifar100',
    'mnist',
    'fashion',
    'ldpa',
]


for ds in datasets:
    # Load database
    accuracy = pd.read_csv(str(Path(ds) / 'results' / 'accuracy.csv'))

    accuracy = accuracy[accuracy['method'].isin(['iwkmeans', 'wkmeans', 'margin', 'uncertainty', 'iconfidence', 'random'])]

    accuracy['method'] = accuracy['method'].replace(names)

    # compute auc
    accuracy = accuracy.drop_duplicates(subset=['dataset', 'method', 'seed', 'n_iter'])
    f = lambda x: auc(x['n_iter'], x['value'])
    accuracy = accuracy.groupby(['dataset', 'method', 'seed']).apply(f)

    df = accuracy.unstack(['method']).reset_index()
    df = df.drop(['dataset', 'seed'], axis=1)
    df.columns.name = None

    result = autorank(df)

    plot_stats(result)
    plt.savefig('method_ranking_{}.pdf'.format(ds))