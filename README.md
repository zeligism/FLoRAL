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

## Using and tuning FLoRAL
FLoRAL learns a shared base model and multiple shared sets of adaptors (the clusters).
Each client learns a private router that mixes these adaptors for personalization.
After activating the environment, run a short example from the repository root:
```bash
python -m floral.main \
  task@_global_=synthetic_linear method@_global_=floral \
  floral.num_clusters=2 router_temp=1.0 \
  num_rounds=100 experiment=floral_tuning identifier=k2_t1
```
Choose other tasks from `floral/conf/task/` and increase `num_rounds` for longer training.
The main FLoRAL options are in `floral/conf/method/floral.yaml`;
shared router options are in `floral/conf/method/_method_with_router.yaml`.

### Number of clusters
Set `floral.num_clusters` to choose the number of adaptor sets, not the number of clients.
In the base FLoRAL preset, it defaults to `dataset.num_clusters` when available, otherwise `4`;
task and method presets can override it.
The effective count is `round(floral.num_clusters * floral.num_clusters_mult)`,
where the multiplier defaults to `1.0`.
Keep the multiplier at `1.0` when comparing explicit cluster counts.
Override `floral.num_clusters`, rather than `dataset.num_clusters`, to tune model capacity
without changing the data generation or partitioning.

Use a positive count for an adaptor mixture.
One cluster gives a shared-adaptor baseline with no routing choice.
More clusters allow greater specialization but increase adaptor parameters, communication, and computation.
Start with a small count, such as `2` or `4`, and compare against `1` before increasing it.

### Router temperature
Set the top-level `router_temp` option (default `1.0`), which populates `floral.router_opts.temp`.
The router computes softmax probabilities from its logits divided by this temperature.
For fixed logits:

| Temperature | Effect |
| --- | --- |
| `0.2` | Sharper probabilities, favoring fewer adaptors. |
| `1.0` | Default starting point. |
| `5.0` | Smoother probabilities, mixing adaptors more evenly. |

Keep the temperature strictly positive.
Lowering it sharpens the probabilities but does not itself enable hard routing.
The same temperature is used during training and evaluation.
During training only, `router_noise_std` (default `1.0`) controls additive noise applied before temperature scaling.
The router learning rate is controlled separately by `router_lr`, which defaults to `lr`
unless overridden by a task preset.
Keep these controls fixed initially; if routing is unstable, try reducing the noise or router learning rate.

### Comparing settings
Try cluster counts such as `1`, `2`, and `4` with temperatures such as `0.2`, `1.0`, and `5.0`.
Give each single run a distinct `identifier` (for example, `identifier=k4_t02`) to keep its outputs separate.
Compare held-out validation performance using the same task, training budget, and seeds:
more clusters or a lower temperature are not automatically better.
Append `--cfg job --resolve` to a command to inspect its resolved configuration without training.
See [Visualizing the results](#visualizing-the-results) for tools to inspect experiment results.

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