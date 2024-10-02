# FLoRAL

This is the official code repository for the paper: "Collaborative and Efficient Personalization with Mixtures of Adaptors."
FLoRAL stands for "Federated Low-Rank Adaptive Learning", which is inspired from Low Rank Adaptors or LoRAs.
Later in the project, we started to focus on mixtures of parameter-efficient adaptors instead of only low rank adaptors,
so, in general, adaptors do not necessarily have to be "low rank".
Still, we maintain the name because it sounds cool.

# Reproducing the results
This section should help you get started.

## Create the environment
First, please install the dependencies in a conda environment
```
conda env create -f environment.yaml
```
The new environment will be called floral and can be activated as `conda activate floral`.
Hereafter, we will assume that the enviroment has already been activated.

If the enviroment creation doesn't work, takes too long, or causes other problems
this will likely be due to `tensorflow` or `tensorflow-federated`,
So, just comment them out from `environment.yaml` and don't run the experiments that uses Tensorflow Federated
(namely, Shakespeare, EMNIST, and Stack Overflow).
Sorry, but I don't have a solution to work around `tensorflow-federated`'s dependency resolver right now.

## Testing the environment/experiment
We use OmegaConf style configuration files, so the arguments can be hierarchical.
We additionally make use of Hydra's instantiation mechanism
so that objects can be fully (or patially) declared from the config files.

If the user wants to test the enviroment or a particular experiment,
then they can either edit the `scripts/test.py` script and simply run
```
python scripts/test.py
```
or they can directly run
```
python -m floral.main [<OVERRIDES>]
```
where `<OVERRIDES>` denotes a list of arguments overridden from the command line (e.g., `lr=1e-3` or `method@_global_=ensemble`).
You can always edit the `floral/conf/base.yaml` directly,
but we recommend overriding it using the test script `scripts/test.py`.
Note that the two arguments `method@_global_` and `task@_global_` have the `@_global_` suffix
because they are "packages" arguments that import many other prespecified arguments to the global level.

## Running the experiments
I prepared a nice script for running experiments on SLURM automatically, if there is any.
Otherwise, it will run sequentially on your local machine.
From the working directory, run the following command
```
python scripts/run_experiment.py [<NAME_OF_EXPERIMENT>] [<NAME_OF_DATASET>...]
```
Both arguments are optional.
The first argument is the name of the experiment, and the rest are datasets/tasks.
If empty, a list of available experiments will be shown, and you
will be prompted to choose one experiment from the list.
If the first argument is provided and the second is empty,
then you will be provided a list of possible experiments,
from which you can choose one or more.

SLURM will be detected automatically.
If not detected, the experiments will be run locally in sequence.
Otherwise, a submitit launcher config will be checked in `floral/conf/hydra/launcher/`,
where the launcher config should share the name of the given dataset without subtask suffixes
(please see the launcher directory for examples of launcher configs, which can be edited as desired).
If the launcher config doesn't exist, you will be notified and asked whether to run locally or terminate.
Otherwise, jobs running the experiments will be submitted automatically.
(Note: the main scripts will run with the option `overwrite_sweep` set to false by default (see `floral/conf/base.yaml`),
so if the result of another experiment with the same sweep argument is found, the script will terminate gracefully,
so make sure you delete or move the previous results when running a fresh batch of experiments.)

## Visualizing the results
We provide two notebooks for visualizing the results:
`notebooks/plot_experiments.ipynb` and `notebooks/generate_latex_tables.ipynb`.
The first notebook plots the losses and other metrics of the experiments you ran
(all variations are provided and a choice can be made with commenting/uncommenting).
It will also generate a `metrics.csv` file,
which is a table of the final performances of all the variations of experimental variables of interest.
The second notebook generate latex tables from the `metrics.csv` generated earlier.

We also provide a simple script if the user want to generate all the plots for all the experiments
```
python -m floral.plot [<NAME_OF_EXPERIMENT>]
```
The argument list is a list of specific experiments for which you want the plots to be generated.
If empty, plots for all known experiments will be generated.
The user will be notified for failed experiments and will be shown the error traceback at the end.