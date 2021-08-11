import pandas as pd
from matplotlib import pyplot as plt
import os
import matplotlib
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path

import argparse
import importlib


import sys
sys.path.append("../")
sys.path.append('../exp/')
from study import init_figure, plot_by_method, save_figure, plot_boxplot, plot_table, names, namify


parser = argparse.ArgumentParser()
parser.add_argument('--out', action="store", default='screen', choices=['screen', 'png', 'pdf'], help='Output destination')
parser.add_argument('-m', action="store", nargs='*', default=[], help='Metrics to print')
parser.add_argument('methods', nargs='*', help='Methods to print')

args = parser.parse_args()


output = args.out
metrics = args.m
methods = args.methods

del args

def output_figure(out, dataset, metric, legend_kwargs={}):

    if legend_kwargs is not None:
        plt.legend(**legend_kwargs)
    plt.xlabel('Training sample count')
    plt.ylabel(names.get(metric, namify(metric)))
    
    if out == 'screen':
        plt.show()
    elif out == 'png':
        plt.savefig('{}{}_{}.png'.format(dataset, suffix, metric), bbox_inches='tight', pad_inches=0)
        plt.close()
    elif out == 'pdf':
        plt.savefig('{}{}_{}.png'.format(dataset, suffix, metric), bbox_inches='tight', pad_inches=0)
        plt.close()


font = {}
font['size'] = 16


matplotlib.rc('font', **font)
plt.rcParams["mathtext.fontset"] = "stix"

ds = Path.cwd().name
res_path = Path('results/')

exp_module = importlib.import_module(ds)
exp_config = exp_module.get_config()

start_size = exp_config['start_size']
batches = exp_config['batches']
samples = np.cumsum([start_size] + batches)

for name in metrics:
    csv_path = res_path / (name + '.csv')

    if not csv_path.exists():
        print('{} not found'.format(str(csv_path)))
        continue

    print('Plotting ', name)
    df = pd.read_csv(str(csv_path))

    if df['value'].dtype == object:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[~df['value'].isna()]
    
    if not 'n_samples' in df.columns:
        df['n_samples'] = samples[df['n_iter']]

    if len(methods) > 0:
        df = df[df['method'].isin(methods)]

    if name == 'hard_contradiction':
        df['value'] = 1 - df['value']
    
    init_figure()
    print(df) 
    plot_by_method(df, log=(name == 'top_exploration'))
    
    legend_kwargs = {}
    if name == "accuracy":
        legend_kwargs['loc'] = 4
    if name == "top_exploration":
        pass
        #plt.xscale('log')
        #plt.xticks(iters[ds][1:])
        #plt.gca().get_xaxis().get_major_formatter().labelOnlyBase = False

    if name in ['accuracy', 'hard_contradiction', 'batch_agreement', 'batch_difficulty']:
        formatter = ticker.FormatStrFormatter('%.2f')
        plt.gca().yaxis.set_major_formatter(formatter)
    
    if name in ['batch_difficulty']:
        output_figure(output, ds, name, legend_kwargs=None)

    output_figure(output, ds, name, legend_kwargs=legend_kwargs)

    # if name == 'accuracy':
    #     plot_table(df, cumsum=True)
    # if name == 'noisy_in_selected':
    #     plot_table(df, last=True, normalize=True)
    # if name == 'batch_reverse_accuracy':
    #     plot_table(df, cumsum=True)
    # if name == 'batch_reverse_score':
    #     plot_table(df, cumsum=True)

    plt.close()
