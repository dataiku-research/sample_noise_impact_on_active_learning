# Sample Noise Impact on Active Learning

## Prerequisites

This package requires the cardinal python package

`pip install cardinal`

## Generate accuracy figures and tables

In order to generate a figure for a given task, simply go in the directory and call the plotting script.
The script has some options but by default it generates the figures of the paper.

```sh
cd cifar10
python ../plots.py
```

## Re-run the experiments

In order to run an experiment, simply go in the folder of the corresponding task and call the run script:

```sh
cd cifar10
python ../exp/run.py
```

One can specify the query sampling method to run as an argument to the script bur by default it run the experiments of the paper.
