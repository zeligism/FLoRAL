# You can profile a client process from shell with py-spy:
# >> py-spy top --pid <worker_pid>
# or generate a flame graph
# >> py-spy record -o profile.svg --pid <worker_pid>

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

# task = "synthetic_linear"
# task = "synthetic_mlp"
# task = "synthetic_mlp_bn"
# task = "cifar10_label_shift"
task = "mnist_label_shift_reduced"
# task = "mnist_rotate"
# task = "cifar100"
# task = "emnist"
# task = "shakespeare"
# task = "stackoverflow"


def remove_suffixes(task: str):
    # Remove subtask suffix, if any
    for suffix in SUBTASK_SUFFIXES:
        task = task.removesuffix(suffix)
    return task


if __name__ == "__main__":
    command_main = ["python", "-m", "floral.main"]
    command_launcher = [] if not LAUNCH else [
        "--multirun", f"hydra/launcher=submitit_{remove_suffixes(task)}",
    ]
    command_args = [
        f"experiment=test_{task}",
        f"task@_global_={task}",
        f"method@_global_=floral",
        f"identifier=test",
        f"num_rounds=10",
        f"+ray_init_args.num_cpus=32",

        # f"experiment=test_{task}",
        # f"task@_global_={task}",
        # # f"method@_global_=floral",
        # f"method@_global_=floral_optimalrouter",
        # f"identifier=whydoesfloralon{task}suck_optimal",
        # f"num_rounds=10",
        # f"+ray_init_args.num_cpus=10",
        # "max_logfiles=10000",
        # "precond_eps=1e-5",

        # This has great performance
        # "model.batch_norm=True",
        # "floral.use_normlora=True",
        # "floral.normlora_reparam=True",
        # "extras@_global_=local_batchnorm",
    ]
    subprocess.call(command_main + command_launcher + command_args)
