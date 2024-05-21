# Utility script for running experiments using hydra sweepers.
# The command should be given as follows:
# >> python scripts/run_experiment.py [<EXPERIMENT>] [<TASKS>...]
# <EXPERIMENT> is a sweeper config file name, defined in EXPERIMENT_CONFIG_DIR.
# <TASKS>... is a space-separated list of tasks, defined in TASK_CONFIG_DIR.
# 
# The first argument is the name of the experiment, and the rest are tasks.
# If empty, a list of available experiments (tasks) will be shown, and you
# will be prompted to choose one experiment (one or more tasks) from the list.
# 
# If slurm is not detected, the program will proceed normally and run locally.
# Otherwise, a submitit launcher config will be checked in LAUNCHER_CONFIG_DIR,
# which should share the name of the given task without subtask suffixes, and
# if the launcher doesn't exist, you will be asked to run locally or terminate.
# 
# Suggested Experiments:
# - python scripts/run_experiment.py run_methods synthetic_linear synthetic_mlp mnist_rotate mnist_label_shift \
#                                                   cifar10_rotate cifar10_label_shift cifar100 emnist shakespeare stackoverflow
# - python scripts/run_experiment.py ab_floral cifar10_rotate cifar10_label_shift cifar100 emnist
# - python scripts/run_experiment.py ab_normlora emnist stackoverflow
# - python scripts/run_experiment.py hp_floral cifar10_rotate cifar10_label_shift cifar100 emnist shakespeare
# - python scripts/run_experiment.py hp_convlora cifar10_rotate cifar10_label_shift cifar100 emnist
# - python scripts/run_experiment.py hp_batchnormlora synthetic_mlp_bn cifar100
# 

import sys
import os
import glob
import shutil
import subprocess
import time

EXPERIMENT_CONFIG_DIR = "floral/conf/hydra/sweeper"
TASK_CONFIG_DIR = "floral/conf/task"
LAUNCHER_CONFIG_DIR = "floral/conf/hydra/launcher"
SWEEP_ID = "sweep"
SUBTASK_SUFFIXES = (
    "_simple",
    "_rotate",
    "_label_shift",
    "_reduced",
    "_bn",
)
WAIT = False


def get_config_names(conf_dir):
    config_names = []
    for fname in glob.glob(os.path.join(conf_dir, "*")):
        config_name = os.path.basename(fname).replace('.yaml', '')
        if ".yaml" in fname and not config_name.startswith("_"):
            config_names.append(config_name)
    return list(sorted(config_names))


def remove_suffixes(task: str):
    # Remove subtask suffix, if any
    for _ in range(2):  # removes two combined suffixes
        for suffix in SUBTASK_SUFFIXES:
            task = task.removesuffix(suffix)
    return task


def parse_args():
    args = sys.argv[1:]
    available_experiments = get_config_names(EXPERIMENT_CONFIG_DIR)
    available_tasks = get_config_names(TASK_CONFIG_DIR)
    experiment = args[0] if len(args) > 0 else input(
        f"Which experiment would you like to run? ({' / '.join(available_experiments)}): "
    )
    tasks = args[1:] if len(args) > 1 else input(
        f"Which tasks would you like to run? ({' / '.join(available_tasks)}): "
    ).split()
    # check for user erros and typos
    if experiment not in available_experiments:
        raise ValueError(f"Experiment '{experiment}' is not among {available_experiments}.")
    for task in tasks:
        if task not in available_tasks:
            raise ValueError(f"Task '{task}' is not among {available_tasks}.")

    return experiment, tasks


if __name__ == "__main__":
    ps = {}
    experiment, tasks = parse_args()
    slurm_detected = shutil.which("srun") is not None
    if slurm_detected:
        print("Slurm detected. Submitting jobs with submitit.")
    print(f"Running experiment '{experiment}' on the following tasks: {tasks}")
    named_commands = {}
    for task in tasks:
        command = [
            "python", "-m", "floral.main", "--multirun",
            f"hydra/sweeper={experiment}",
            f"experiment={experiment}_{task}",
            f"task@_global_={task}",
            f"identifier={SWEEP_ID}",
            # TODO: uncomment this when code is published
            # "loglevel=INFO",
            # This helps construct a nice dirname for the sweeps
            "hydra.job.config.override_dirname.exclude_keys=[experiment,identifier]",
        ]

        # Use slurm launcher if slurm is detected
        if slurm_detected:
            # A launcher for some task would be the same for all its subtasks
            launcher = f"submitit_{remove_suffixes(task)}"
            available_launchers = get_config_names(LAUNCHER_CONFIG_DIR)
            if launcher in available_launchers:
                command += [f"hydra/launcher={launcher}"]
            else:
                print(f"Couldn't find a submitit launcher config file '{launcher}' for task '{task}'.")
                if "n" == input("Run locally without slurm? [y]/n: "):
                    sys.exit("Terminating program.")
        else:
            print("Slurm not detected. Running locally.")

        named_commands[f"{experiment}_{task}"] = command

    # Run
    for i, (command_name, command) in enumerate(named_commands.items()):
        print(f"\n({i+1}) Running '{command_name}':", " ".join(command))
        ps[task] = subprocess.Popen(command)
        time.sleep(1)  # take it easy

    if WAIT:
        # wait for all sweep processes to finish (not necessary)
        print("Waiting for processes to finish...")
        exit_codes = {k: p.wait() for k, p in ps.items()}
        print(f"Finished with exit codes: {exit_codes}")
