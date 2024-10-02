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
#                                        cifar10_rotate cifar10_label_shift cifar100 emnist shakespeare stackoverflow
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
# import psutil
import subprocess
import time


EXPERIMENT_CONFIG_DIR = "floral/conf/hydra/sweeper"
TASK_CONFIG_DIR = "floral/conf/task"
LAUNCHER_CONFIG_DIR = "floral/conf/hydra/launcher"
DEFAULT_LOCAL_LAUNCHER = "joblib"
SWEEP_ID = "sweep"
SUBTASK_SUFFIXES = (
    "_simple",
    "_rotate",
    "_label_shift",
    "_reduced",
    "_bn",
)

# Only if you have a lot of cpus locally, run 'FORCE_RUN_LOCALLY=1 python run_exper...'
FORCE_RUN_LOCALLY = os.environ.get("FORCE_RUN_LOCALLY") is not None
MAX_N_JOBS_LOCALLY = os.environ.get("MAX_N_JOBS_LOCALLY", 8)
CPUS_PER_JOB_LOCALLY = os.environ.get("CPUS_PER_JOB_LOCALLY", 8)  # roughly speaking
GPUS_PER_JOB_LOCALLY = os.environ.get("GPUS_PER_JOB_LOCALLY", 0)

_printed = set()


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


def print_once(s):
    global _printed
    if s not in _printed:
        print(s)
        _printed.add(s)


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
    experiment, tasks = parse_args()
    slurm_detected = shutil.which("srun") is not None
    print("Slurm detected." if slurm_detected else "Slurm not detected.")
    should_use_slurm = slurm_detected and not FORCE_RUN_LOCALLY
    print("Submitting slurm jobs with submitit." if should_use_slurm else "Ignoring slurm. Running locally.")

    print(f"Running experiment '{experiment}' on the following tasks: {tasks}")
    named_commands = {}
    for task in tasks:
        command = [
            "python", "-m", "floral.main", "--multirun",
            f"hydra/sweeper={experiment}",
            f"experiment={experiment}_{task}",
            f"task@_global_={task}",
            f"identifier={SWEEP_ID}",
            "continue_training=True",
            # TODO: uncomment this when code is published
            # "loglevel=INFO",
        ]
        exclude_keys = ["experiment", "identifier", "continue_training"]  # exclude them from sweep dir name

        # Use slurm launcher if slurm is detected
        found_submitit_launcher = False
        if should_use_slurm:
            # A launcher for some task would be the same for all its subtasks
            launcher = f"submitit_{remove_suffixes(task)}"
            available_launchers = get_config_names(LAUNCHER_CONFIG_DIR)
            if launcher in available_launchers:
                found_submitit_launcher = True
                command += [f"hydra/launcher={launcher}"]
            else:
                print(f"Couldn't find submitit launcher config '{launcher}' for task '{task}'.")

        # Cannot submitit to slurm :(
        if not found_submitit_launcher:
            # Check capacity for running locally
            total_cpus = os.cpu_count() or CPUS_PER_JOB_LOCALLY
            n_jobs = min(MAX_N_JOBS_LOCALLY, max(1, total_cpus // CPUS_PER_JOB_LOCALLY))

            if int(GPUS_PER_JOB_LOCALLY) > 0:
                print_once("Running on gpu.")
                command += ["hydra.sweeper.n_jobs=1"]
                command += [f"+ray_init_args.num_cpus={CPUS_PER_JOB_LOCALLY}"]
                command += [f"client_resources.num_cpus={int(CPUS_PER_JOB_LOCALLY) // int(GPUS_PER_JOB_LOCALLY)}"]
                command += ["client_resources.num_gpus=1"]
                exclude_keys.append("ray_init_args.num_cpus")
                exclude_keys.append("client_resources.num_cpus")
                exclude_keys.append("client_resources.num_gpus")
            elif n_jobs == 1:
                print_once("Capacity of local machine is not great. Running jobs sequentially (too slow).")
                command += ["hydra.sweeper.n_jobs=1"]
                command += ["dataloader.num_workers=0"]
                exclude_keys.append("dataloader.num_workers")
            else:
                print_once("Capacity of local machine is ok.")
                print_once(f"Will launch {n_jobs} jobs in parallel using {DEFAULT_LOCAL_LAUNCHER}.")
                command += [f"hydra/launcher={DEFAULT_LOCAL_LAUNCHER}"]
                command += [f"hydra.sweeper.n_jobs={n_jobs}"]
                command += [f"+ray_init_args.num_cpus={CPUS_PER_JOB_LOCALLY}"]
                exclude_keys.append("ray_init_args.num_cpus")

        command += [f"hydra.job.config.override_dirname.exclude_keys=[{','.join(exclude_keys)}]"]
        named_commands[f"{experiment}_{task}"] = command

    # Run
    ps = {}
    exit_codes = {}
    for i, (command_name, command) in enumerate(named_commands.items()):
        print(f"[{i+1}/{len(named_commands)}] Running '{command_name}':", " ".join(command))
        has_submitit_launcher = any("submitit" in arg for arg in command)
        if has_submitit_launcher:
            # Run in background since it's a light job submission process
            ps[task] = subprocess.Popen(command)
            time.sleep(1)  # take it easy
        else:
            # always run tasks sequentially (each task already includes a few jobs)
            exit_codes[task] = subprocess.call(command)

    # Wait, if any
    try:
        for k, p in ps.items():
            if isinstance(p, subprocess.Popen):
                print(f"Waiting for {k} to finish...")
                exit_codes[k] = p.wait()
    except:
        for k, p in ps.items():
            if isinstance(p, subprocess.Popen):
                p.terminate()
        raise
    print(f"Finished all processes with exit codes: {exit_codes}")
