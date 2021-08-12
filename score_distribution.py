import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from study import names
from pathlib import Path
import itertools
from cardinal.utils import ActiveLearningSplitter
import numpy as np
from study import names
import pandas as pd


datasets = [
    'cifar10',
    'cifar10_simclr',
    'cifar100',
    'mnist',
    'fashion',
    'ldpa',
]


preds_per_ds = {}

for ds in datasets:

    preds = []

    for seed, flip in itertools.product(['11', '22', '33', '44', '55'], ['A', 'B']):

        proba = np.load(str(Path(ds) / 'precomputed_proba' / (seed + flip) / 'proba.npy'))

        splitter = ActiveLearningSplitter(proba.shape[0], test_size=.5, random_state=int(seed))

        if flip == 'B':
            # Exchange -1 and -2
            splitter._mask = -(splitter._mask + 2) - 1

        preds.append(proba[splitter.train])

    preds_per_ds[ds] = np.hstack(preds)

    # plt.hist(np.hstack(preds), bins=50)
    # plt.title(names.get(ds))
    # plt.show()

labels, data = [*zip(*preds_per_ds.items())]  # 'transpose' items to parallel key, value lists

# plt.boxplot(data)
p = plt.violinplot(data, showextrema=False, widths=.6)

for pc in p['bodies']:
    pc.set_edgecolor('black')
    pc.set_linewidth(1)
    pc.set_alpha(0.5)

plt.xticks(range(1, len(labels) + 1), [names.get(l, l) for l in labels], rotation=20)

# for i, l in enumerate(labels):
#     y = data[i]
#     # Add some random "jitter" to the x-axis
#     x = np.random.normal(i, 0.05, size=len(y))
#     plt.plot(x, y, 'r.', alpha=0.01)

plt.savefig('score_distribution.pdf', bbox_inches='tight', pad_inches=0)