
import os
import glob
import pickle
import traceback
import pandas as pd
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from omegaconf import OmegaConf
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.figsize'] = (5, 3)  # use for publication
# mpl.rcParams['figure.figsize'] = (9, 6)
LEGEND_SIZE = 9
LEGEND_NCOL = 3

BASE_DIR = os.environ.get("OUTPUT_DIR", ".")
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FILE_PATTERN = "*/*/*.pkl"  # <experiment>/<task_dir>/history.pkl
DEFAULT_DOWNSAMPLE_LEN = 50  # rounds
DEFAULT_TIME_PERIOD = 60  # seconds

MODES = ["centralized", "distributed", "distributed_fit"]
INDICES = ["round", "time"]
METRICS = [
    "loss", "loss_in_vocab",
    "acc", "accuracy",
    "accuracy_top1", "accuracy_top3",
    "accuracy_top5", "accuracy_top10",
    "router_entropy", "W_error", "uv_error",
]

VARIABLE_TO_LATEX = {
    "method": "Method",
    "optimal_router": r"Optimal $\pi$",
}
VARIABLE_FIELDS_TO_LATEX = {
    "method": {
        "ensemble": "Ensemble",
        "fedavg": "FedAvg",
        "floral": "FLoRAL(1%)",
        "floral_optimalrouter": "FLoRAL(1%)",
        "floral_10": "FLoRAL(10%)",
        "floral_10_optimalrouter": "FLoRAL(10%)",
        "locallora": "Local Adaptor",
        "fedprox": "FedProx",
        "ditto": "Ditto",
    },
    "optimal_router": {
        False: False,
        True: True,
    },
}

AVAILABLE_EXPERIMENTS = [
    # ----- Methods ------ #
    # Clustered datasets
    "run_methods_synthetic_linear",
    "run_methods_synthetic_mlp",
    "run_methods_mnist_rotate",
    "run_methods_mnist_label_shift",
    "run_methods_cifar10_rotate",
    "run_methods_cifar10_label_shift",
    "run_methods_cifar100",
    "run_methods_mnist_rotate_reduced",
    "run_methods_mnist_label_shift_reduced",
    "run_methods_cifar10_rotate_reduced",
    "run_methods_cifar10_label_shift_reduced",
    "run_methods_cifar100_reduced",
    # General datasets
    "run_methods_emnist",
    "run_methods_shakespeare",
    "run_methods_stackoverflow",
    # ----- Ablation ----- #
    # Ablate LoRAs, ConvLoRAs and bias
    "ab_floral_cifar10_rotate",
    "ab_floral_cifar10_label_shift",
    "ab_floral_cifar100",
    "ab_floral_emnist",
    # Ablate NorAs separately (informed by results from ab_floral)
    "ab_normlora_cifar100",
    "ab_normlora_emnist",
    "ab_normlora_stackoverflow",
    # ----- Hyperparameters ----- #
    # FLoRAL num clusters and rank
    "hp_floral_cifar10_rotate",
    "hp_floral_cifar10_label_shift",
    "hp_floral_cifar100",
    "hp_floral_emnist",
    "hp_floral_shakespeare",
    # ConvLoRA methods
    "hp_convlora_cifar10_rotate",
    "hp_convlora_cifar10_label_shift",
    "hp_convlora_cifar100",
    "hp_convlora_emnist",
    # Batchnorm methods
    "hp_batchnormlora_synthetic_mlp_bn",
    "hp_batchnormlora_cifar100_bn",
    # Router regularization
    "hp_regularizer_cifar10_rotate",
    "hp_regularizer_cifar10_label_shift",
]


def load_runs(output_dir=OUTPUT_DIR):
    histories = {}
    for run in glob.glob(os.path.join(output_dir, FILE_PATTERN)):
        try:
            with open(run, "rb") as f:
                history = pickle.load(f)
        except:
            continue
        histories[run] = history
    return histories


def flwr_history_to_df(history):
    if isinstance(history, dict):
        try:
            # Version 2
            history_dict = {
                "loss_centralized": history["losses_centralized"],
                "loss_distributed": history["losses_distributed"],
                **{k+"_centralized": v for k, v in history["metrics_centralized"].items() if k != "round"},
                **{k+"_distributed": v for k, v in history["metrics_distributed"].items() if k != "round"},
                **{k+"_distributed_fit": v for k, v in history["metrics_distributed_fit"].items() if k != "round"},
            }
        except KeyError:
            # Version 3 (TODO: deprecate versions 1 and 2)
            history_dict = history
    else:
        # Version 1
        history_dict = {
            "loss_centralized": history.losses_centralized,
            "loss_distributed": history.losses_distributed,
            **{k+"_centralized": v for k, v in history.metrics_centralized.items() if k != "round"},
            **{k+"_distributed": v for k, v in history.metrics_distributed.items() if k != "round"},
            **{k+"_distributed_fit": v for k, v in history.metrics_distributed_fit.items() if k != "round"},
        }

    # history_dict = {"Loss": history.losses_distributed}
    # for metric_name, metric_history in history.metrics_distributed.items():
    #     if metric_name == "round":
    #         continue
    #     if "acc" in metric_name and "accuracy" not in metric_name:
    #         metric_name = metric_name.replace("acc", "accuracy")
    #     history_dict[metric_name] = metric_history
    # for metric_name, metric_history in history.metrics_distributed_fit.items():
    #     if metric_name == "round":
    #         continue
    #     if "acc" in metric_name and "accuracy" not in metric_name:
    #         metric_name = metric_name.replace("acc", "accuracy")
    #     history_dict["Train "+metric_name] = metric_history

    history_df_list = [
        pd.DataFrame(history_dict[k], columns=["round", k]).set_index("round")
        for k in history_dict.keys() if len(history_dict[k]) > 0
    ]
    assert len(history_df_list) > 0
    history_df = pd.concat(history_df_list, axis="columns").reset_index()
    # add 'time' from 'duration
    for mode in ["centralized", "distributed", "distributed_fit"]:
        if f"duration_{mode}" in history_df.columns:
            history_df[f"time_{mode}"] = history_df[f"duration_{mode}"].fillna(0.).cumsum()
            history_df = history_df.drop(columns=[f"duration_{mode}"])
        if f"time_{mode}" in history_df.columns:
            history_df[f"time_{mode}"] -= history_df[f"time_{mode}"].min()
    return history_df


def histories_to_df(
        histories,
        filter_values=None,
        ignore_values=None,
        downsample_index=INDICES[0],
        downsample_len=DEFAULT_DOWNSAMPLE_LEN,
        hide_na=True,
        ):
    if len(histories) == 0:
        print("Histories list is empty!")
        return pd.DataFrame()

    df_list = []
    for run, history in histories.items():
        cfg = history["cfg"]
        if not cfg_satisfies(cfg, filter_values, should_intersect=True):
            continue
        if not cfg_satisfies(cfg, ignore_values, should_intersect=False):
            continue
        try:
            df = pd.DataFrame(data=flwr_history_to_df(history["history"]))
        except AssertionError:
            print(f"Encountered a problem with run '{run}'")
            continue
        if len(df) == 0:
            print(f"Run '{run}' is empty!")
            continue
        if len(df) == 1:
            print(f"Run '{run}' consists of a single round!")
            continue
        if hide_na and len(df["loss_distributed"]) > 0 and pd.isna(df["loss_distributed"].iloc[-1]):
            print(f"Run '{run}' final loss is NaN!")
            continue
        # This downsampling assumes a df of metrics for one run only
        df = downsample_to_len(df, downsample_len, index=downsample_index)
        df = add_cfg_to_df(cfg, df)
        df_list.append(df)

    if len(df_list) == 0:
        print("Could not find any run that satisfies the given filter configuration.")
        return pd.DataFrame()

    return pd.concat(df_list).reset_index(drop=True)


def cfg_satisfies(source, target, should_intersect=True, root="/", verbose=False):
    if target is None:
        return True
    # Root case: target is a list of values, and source is
    # a value that should (not) be contained in target
    if isinstance(target, ListConfig):
        if should_intersect and source not in target:
            if verbose:
                print(f"{root} FAILED: source value {source} is not found in target {target}")
            return False
        elif not should_intersect and source in target:
            if verbose:
                print(f"{root} FAILED: source value {source} is found in target {target}")
            return False
    elif isinstance(target, DictConfig):
        for key in target.keys():
            if key not in source:
                # impossible to satisfy if key does not exist
                if verbose:
                    print(f"{root} FAILED: Couldn't find {key} in source")
                return False
            else:
                if verbose:
                    print(f"{root}: Recursively satisfying {root}{key}")
                satisfied = cfg_satisfies(source[key], target[key],
                                          should_intersect=should_intersect,
                                          root=f"{root}{key}/",
                                          verbose=verbose)
                if not satisfied:
                    return False
    else:
        if verbose:
            print(f"{root} FAILED: target leaf node is not a list")
        return False

    # all (non-)intersection constraints were satisfied
    if verbose:
        print(f"{root}: OK")
    return True


def downsample_to_len(
        df,
        downsample_len=DEFAULT_DOWNSAMPLE_LEN,
        index="round",
        time_period=DEFAULT_TIME_PERIOD,
        ):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    other_cols = df.select_dtypes(exclude="number").columns.tolist()
    period = (df[index].max() - df[index].min()) / downsample_len

    # Creating an index of downsampled rounds with mean values (dropping the older round col)
    downsampled_df = df.groupby(df[index].apply(lambda i: round(period * round(i / period))))
    # Take the mean of the numeric columns
    downsampled_df = downsampled_df[numeric_cols].mean().drop(columns=[index])
    # Values less than 1e-20 are effectively 0 (numeric cols only)
    downsampled_df[downsampled_df < 1e-20] = 0.0
    # Re-assign non-numeric columns
    if len(other_cols) > 0:
        downsampled_df[other_cols] = df.set_index(index)[other_cols]
    # Make index into column
    downsampled_df = downsampled_df.reset_index()
    for time_index in filter(lambda c: "time" in c, downsampled_df.columns):
        downsampled_df[time_index] = time_period * round(downsampled_df[time_index] / time_period)

    return downsampled_df


def add_cfg_to_df(cfg, df):
    df["identifier"] = cfg.identifier
    df["task"] = cfg.task
    df["method"] = cfg.method
    if "extras" in cfg:  # TODO: remove
        df["extras"] = cfg.extras
    df["seed"] = cfg.seed
    df["optimizer"] = cfg.optimizer._target_.split('.')[-1]
    df["lr"] = cfg.lr
    df["weight_decay"] = cfg.weight_decay
    for name, reg in cfg.regularizer.regularizers.items():
        df[name] = reg.parameter
    if cfg.method.startswith("floral"):
        df["router_lr"] = cfg.router_lr
        df["router_entropy"] = cfg.router_entropy
        df["optimal_router"] = cfg.router_diagonal_init
        df["rank"] = cfg.floral.rank
        df["alpha"] = cfg.floral.alpha
        df["num_clusters"] = int(cfg.floral.num_clusters_mult * cfg.floral.num_clusters)
        df["router_temp"] = cfg.floral.router_opts.temp
        from omegaconf.errors import ConfigAttributeError
        try:  # TODO: remove
            df["bias"] = cfg.floral.bias
            df["lora"] = cfg.floral.use_linearlora
            df["convlora"] = cfg.floral.use_convlora
            df["embeddinglora"] = cfg.floral.use_embeddinglora
            df["normlora"] = cfg.floral.use_normlora
            df["convlora_method"] = cfg.floral.convlora_method
            df["normlora_reparam"] = cfg.floral.normlora_reparam
        except ConfigAttributeError:
            pass
    elif cfg.method.startswith("ensemble"):
        df["router_lr"] = cfg.router_lr
        df["optimal_router"] = cfg.router_diagonal_init
    else:
        df["optimal_router"] = False

    return df


def setup_experiment_plotting_and_variables(history_df, experiment):
    assert experiment in AVAILABLE_EXPERIMENTS
    if "run_methods" in experiment:
        # XXX: floral_locallora -> locallora
        history_df.loc[history_df["method"] == "floral_locallora", "method"] = "locallora"
        # Declare methods, remove if not found
        methods_sorted = ["fedavg", "locallora", "ensemble", "floral", "floral_10"]
        # methods_sorted += ["fedprox", "ditto"]  # XXX
        available_methods = history_df["method"].unique()
        for method in methods_sorted:
            if method not in available_methods and f"{method}_optimalrouter" not in available_methods:
                methods_sorted.remove(method)
        # remove _optimalrouter suffix (use option instead)
        for method in methods_sorted:
            history_df.loc[history_df["method"] == f"{method}_optimalrouter", "method"] = method
        # plotting options and variables
        variables = ["method"]
        if len(history_df["optimal_router"].unique()) > 1:
                variables += ["optimal_router"]
        for variable in variables:
            history_df[variable] = history_df[variable].map(VARIABLE_FIELDS_TO_LATEX[variable])
        history_df = history_df.rename(columns={k: v for k, v in VARIABLE_TO_LATEX.items() if k in variables})
        plot_opts = {"hue": VARIABLE_TO_LATEX["method"],
                     "hue_order": [VARIABLE_FIELDS_TO_LATEX["method"][method] for method in methods_sorted]}
        if "optimal_router" in variables:
            plot_opts["style"] = VARIABLE_TO_LATEX["optimal_router"]
        variables = [VARIABLE_TO_LATEX[v] for v in variables]

    elif "hp_batchnormlora" in experiment:
        # XXX: remove local_batchnorlora and rename others
        history_df["batchnorm_stats"] = history_df["extras"]
        history_df = history_df[history_df["batchnorm_stats"] != "local_batchnormlora"]
        history_df.loc[history_df["batchnorm_stats"] == "local_batchnorm", "batchnorm_stats"] = "local"
        history_df.loc[history_df["batchnorm_stats"] == "none", "batchnorm_stats"] = "federated"
        history_df.loc[history_df["method"] == "floral", "batchnorm_adaptor"] = "none"
        history_df.loc[history_df["method"] == "floral_normlora", "batchnorm_adaptor"] = "regular"
        history_df.loc[history_df["method"] == "floral_normlora_reparam", "batchnorm_adaptor"] = "reparameterized"
        variables = ["batchnorm_adaptor", "batchnorm_stats"]
        plot_opts = {
            "hue": "batchnorm_adaptor",
            "hue_order": list(sorted(history_df["batchnorm_adaptor"].unique())),
            "style": "batchnorm_stats",
            "style_order": list(sorted(history_df["batchnorm_stats"].unique())),
        }

    elif "hp_convlora" in experiment:
        history_df.loc[history_df["convlora"] == False, "convlora_method"] = "none"
        variables = ["convlora_method"]
        plot_opts = {
            "hue": "convlora_method",
            "hue_order": list(sorted(history_df["convlora_method"].unique())),
        }

    elif "ab_floral" in experiment:
        history_df.loc[(history_df["lora"] == True) & (history_df["convlora"] == True), "active_loras"] = "linear+conv"
        history_df.loc[(history_df["lora"] == True) & (history_df["convlora"] == False), "active_loras"] = "linear"
        history_df.loc[(history_df["lora"] == False) & (history_df["convlora"] == True), "active_loras"] = "conv"
        history_df.loc[(history_df["lora"] == False) & (history_df["convlora"] == False), "active_loras"] = "none"
        variables = ["active_loras", "bias"]
        plot_opts = {
            "hue": "active_loras",
            "hue_order": ["none", "linear", "conv", "linear+conv"],
            "style": "bias",
            "style_order": [False, True],
        }

    elif "hp_floral" in experiment:
        variables = ["num_clusters", "rank"]
        plot_opts = {
            "hue": "num_clusters",
            "hue_order": list(sorted(history_df["num_clusters"].unique())),
            "palette": "tab10",
            "style": "rank",
            "style_order": list(sorted(history_df["rank"].unique())),
        }

    elif "hp_regularizer" in experiment:
        variables = ["router_entropy"]
        plot_opts = {
            "hue": "router_entropy",
            "hue_order": list(sorted(history_df["router_entropy"].unique())),
            "palette": "tab10",
        }

    else:
        variables = []
        plot_opts = {"hue": "identifier"}

    return history_df, plot_opts, variables


def plot_and_save(history_df, plot_opts, results_dir, use_median_and_ci=False, close_figure=False):
    for index, metric, mode in product(INDICES, METRICS, MODES):
        plotting_df = history_df.copy()
        x = index if index == "round" else f"{index}_{mode}"
        y = f"{metric}_{mode}"
        if x not in plotting_df.columns or y not in plotting_df.columns:
            continue
        plotting_df = plotting_df.dropna(subset=x)
        if len(plotting_df) == 0 or plotting_df[y].sum() == 0.0:
            continue

        fig, ax = plt.subplots(1)
        if "acc" not in y:
            ax.set_yscale('log')
        if use_median_and_ci:
            # Reporting the median and CI for experiment runs is better than reporting the mean and std.
            # Unfortunately, it is much slower, so only use this when generating the final plots
            sns.lineplot(x=x, y=y, data=plotting_df, ax=ax, estimator="median", errorbar="ci", **plot_opts)
        else:
            sns.lineplot(x=x, y=y, data=plotting_df, ax=ax, estimator="mean", errorbar="sd", **plot_opts)
        # Configure legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(prop={"size": LEGEND_SIZE}, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{y}_given_{x}.pdf"))
        if close_figure:
            plt.close()


def variables_metrics_to_csv(history_df, variables, results_dir, metrics=METRICS, per_metric=False):
    history_df["total_time"] = history_df["time_distributed"] + history_df["time_distributed_fit"]
    x = "round"
    metrics = map(lambda y: f"{y}_distributed", metrics)  # consider distributed metrics only
    metrics = filter(lambda y: y in history_df.columns, metrics)  # remove unavailable metrics
    metrics = list(metrics)
    last_values_df = history_df[history_df[x] == history_df[x].max()]  # get last values
    if per_metric:
        last_values_df_list = []
        for metric in metrics:
            # XXX: should be checked case by case, but usually a minimize objective has 'loss' in it
            minimize = "loss" in metric
            ranked_last_metric_df = last_values_df[
                [*variables, metric, "total_time"]].sort_values(metric, ascending=minimize)
            ranked_last_metric_df.to_csv(os.path.join(results_dir, f"{metric}.csv"))
            last_values_df_list.append(ranked_last_metric_df)
        return last_values_df_list
    else:
        last_values_df = last_values_df[[*variables, *metrics, "total_time"]].sort_values(variables)
        last_values_df.to_csv(os.path.join(results_dir, "metrics.csv"))
        return last_values_df


def generate_plots(experiments=None, output_dir=OUTPUT_DIR, plots_dir=PLOTS_DIR):
    if experiments is None:
        experiments = AVAILABLE_EXPERIMENTS
    else:
        for experiment in experiments:
            assert experiment in AVAILABLE_EXPERIMENTS, f"Experiment '{experiment}' not available."
    print(f"Plots dir: {plots_dir}")
    print(f"Plotting the following experiments: {experiments}")
    histories = load_runs(output_dir=output_dir)
    assert len(histories) > 0, "No history found!"
    error_msgs = {}
    for experiment in experiments:
        try:
            filter_values = f"""
            experiment: [{experiment}]
            """
            ignore_values = """
            """
            history_df = histories_to_df(
                histories,
                filter_values=OmegaConf.create(filter_values),
                ignore_values=OmegaConf.create(ignore_values),
            )
            assert len(history_df) > 0, f"History of '{experiment}' is empty!"

            results_dir = os.path.join(plots_dir, f"{experiment}")
            os.makedirs(results_dir, exist_ok=True)
            history_df, plot_opts, variables = setup_experiment_plotting_and_variables(history_df, experiment)
            print(f"Plotting {experiment} ...")
            plot_and_save(history_df, plot_opts, results_dir, close_figure=True)
            variables_metrics_to_csv(history_df, variables, results_dir)
            print(f"Done.")
        except:
            print("Failed.")
            error_msgs[experiment] = traceback.format_exc()

    for experiment, error_msg in error_msgs.items():
        print(f"\nTraceback for experiment '{experiment}':")
        print(error_msg)


if __name__ == "__main__":
    generate_plots()
