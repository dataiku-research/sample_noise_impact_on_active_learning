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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


datasets = [
    'ldpa',
    'cifar100',
    'cifar10',
    'cifar10_simclr',
    'fashion',
    'mnist',
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


ax = plt.gca()

mnist1 = mpimg.imread('mnist1.png')
mnist2 = mpimg.imread('mnist2.png')
mnist3 = mpimg.imread('mnist3.png')

im1 = OffsetImage(mnist1, zoom=.8)
im1.image.axes = ax
im2 = OffsetImage(mnist2, zoom=.8)
im2.image.axes = ax
im3 = OffsetImage(mnist3, zoom=.8)
im3.image.axes = ax

ab1 = AnnotationBbox(im1, (6., .6), box_alignment=(0., .5),
                     xybox=(6.8, .85),
                     xycoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                     bboxprops=dict(edgecolor='white'),
)
ax.add_artist(ab1)
ab2 = AnnotationBbox(im2, (6., .5), box_alignment=(0., .5),
                     xybox=(6.8, .5),
                     xycoords='data',
                     arrowprops=dict(arrowstyle="->"),
                     bboxprops=dict(edgecolor='white'),
)
ax.add_artist(ab2)
ab3 = AnnotationBbox(im3, (6., .4), box_alignment=(0., .5),
                     xybox=(6.8, .15),
                     xycoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
                     bboxprops=dict(edgecolor='white'),
)
ax.add_artist(ab3)



# for i, l in enumerate(labels):
#     y = data[i]
#     # Add some random "jitter" to the x-axis
#     x = np.random.normal(i, 0.05, size=len(y))
#     plt.plot(x, y, 'r.', alpha=0.01)

plt.savefig('score_distribution.pdf', bbox_inches='tight', pad_inches=0)
# plt.savefig('score_distribution.pdf', pad_inches=0)
# plt.show()