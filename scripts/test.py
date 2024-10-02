# You can profile a client process from shell with py-spy:
# >> py-spy top -p <worker_pid>
# or sample main process and all of its subprocesses and generate a flame graph
# >> py-spy record -sp <main_pid>

import subprocess

LAUNCH = False
SUBTASK_SUFFIXES = (
    "_simple",
    "_rotate",
    "_label_shift",
    "_rotate_reduced",
    "_label_shift_reduced",
    "_bn",
)


def remove_suffixes(task: str):
    # Remove subtask suffix, if any
    for suffix in SUBTASK_SUFFIXES:
        task = task.removesuffix(suffix)
    return task


# task = "synthetic_linear"
# task = "synthetic_mlp"
# task = "synthetic_mlp_bn"
# task = "cifar10_label_shift"
# task = "mnist_label_shift_reduced"
# task = "mnist_rotate_reduced"
task = "cifar100"
# task = "cifar100_reduced"
# task = "emnist"
# task = "shakespeare"
# task = "stackoverflow"


if __name__ == "__main__":
    command_main = ["python", "-m", "floral.main"]
    command_launcher = [] if not LAUNCH else [
        "--multirun", f"hydra/launcher=submitit_{remove_suffixes(task)}",
    ]
    command_args = [
        f"task@_global_={task}",
        f"experiment=test_{task}",
        "identifier=test3_floral",
        "method@_global_=floral_optimalrouter",
        "num_rounds=100",
        "seed=2",
        "lr=0.001",
        # "continue_training=True",
        # "client_resources.num_gpus=1",
        # "+trainer.clip_grad_norm=10.0",
    ]
    subprocess.call(command_main + command_launcher + command_args)


# python scripts/run_experiment.py run_methods synthetic_linear synthetic_mlp mnist_rotate mnist_label_shift cifar10_rotate cifar10_label_shift cifar100 mnist_rotate_reduced mnist_label_shift_reduced cifar10_rotate_reduced cifar10_label_shift_reduced cifar100_reduced shakespeare
