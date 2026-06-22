"""train_agents.py

Train RL agents for HPC scheduling.

Input contract:
  - CLI args specify algorithm, trace, topology, node config, and hyperparameters.
  - Trace file must be a dev split (holdout is rejected).
  - Split metadata is read from data/splits/logs/*.json.

Output contract:
  - trained_model/{name}/selector/{step}.zip checkpoints
  - trained_model/{name}/train_metadata.json reproducibility sidecar
  - logs/run_log.csv manifest entry (run_id, algorithm, masking, seed, split_id, paths)

References:
  - SB3: https://stable-baselines3.readthedocs.io/
  - Gymnasium: https://gymnasium.farama.org/
"""

from __future__ import annotations

import argparse
import random as r
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from checkpoint import SelectorCheckpointCallback
from HPCsim.HPCsim import HPCsim
from utils import (
    ALGORITHMS,
    ArgumentParserWithDefaults,
    build_train_metadata,
    load_split_metadata,
    resolve_algorithm_config,
    validate_not_holdout,
    write_json,
    write_manifest_entry,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN_LAYER = [4096, 2048, 1024]

ACTIVATION_MAP: dict[str, Any] = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "leaky_relu": nn.LeakyReLU,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults(description="Agent Scheduler training.")

    parser.add_argument(
        "--window",
        default=512,
        dest="window_size",
        metavar="WS",
        help="The observable number of jobs.",
        type=int,
    )
    parser.add_argument(
        "--buffer_size",
        default=1_000_000,
        dest="buffer_size",
        metavar="BUFFER_SIZE",
        help="Replay buffer size (used by DQN).",
        type=int,
    )
    parser.add_argument(
        "--tail",
        default=64,
        dest="tail_size",
        metavar="TS",
        help="The tail size of the queue.",
        type=int,
    )
    parser.add_argument(
        "--save_interval",
        default=100000,
        dest="save_interval",
        metavar="SAVE_INTERVAL",
        help="The interval of saving the model after n every timestep.",
        type=int,
    )
    parser.add_argument(
        "--total_saving",
        default=100,
        dest="total_saving",
        metavar="TOTAL_SAVING",
        help="Total saved checkpoints; total steps = total_saving * save_interval.",
        type=int,
    )
    parser.add_argument(
        "--topology",
        default="physical_topology.txt",
        dest="topology_file",
        metavar="TOPOLOGY_FILE",
        help="The topology file of the environment.",
        type=str,
    )
    parser.add_argument(
        "--trace",
        default="splits/physical_job_dev70.tsv",
        dest="trace_file",
        metavar="TRACE_FILE",
        help="The trace file of the environment.",
        type=str,
    )
    parser.add_argument(
        "--node",
        default="nodes.csv",
        dest="node_file",
        metavar="NODE_FILE",
        help="The node file of the environment.",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="unnamed",
        dest="name",
        metavar="NAME",
        help="The name of the model.",
        type=str,
    )
    parser.add_argument(
        "--hidden",
        default=DEFAULT_HIDDEN_LAYER,
        dest="hidden_layer",
        metavar="HIDDEN",
        help="Hidden layer size.",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--gamma",
        default=0.99,
        dest="gamma",
        metavar="GAMMA",
        help="Discount factor gamma.",
        type=float,
    )
    parser.add_argument(
        "--activation_fn",
        default="tanh",
        dest="activation_fn",
        metavar="ACTIVATION_FN",
        help="Activation function.",
        type=str,
        choices=list(ACTIVATION_MAP.keys()),
    )
    parser.add_argument(
        "--algorithm",
        "--algo",
        dest="algorithm",
        metavar="ALGORITHM",
        help="Name of algorithm.",
        choices=ALGORITHMS.keys(),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--use-masking",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="use_masking",
        metavar="USE_MASKING",
        help="Toggle action masking.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        dest="seed",
        metavar="SEED",
        help="Seed of run.",
        type=int,
    )
    parser.add_argument(
        "--split_id",
        default=None,
        dest="split_id",
        metavar="SPLIT_ID",
        help="Split ID to use (e.g., 'physical_job_r70'). Optional; auto-detects if only one split exists.",
        type=str,
    )

    args = parser.parse_args()
    args.split_id = Path(args.split_id).stem if "." in args.split_id else args.split_id
    print(args)
    return args


# ---------------------------------------------------------------------------
# Validation and setup
# ---------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> None:
    validate_not_holdout(args.trace_file, context="train_args", raise_error=True)
    if args.window_size <= 0 or args.tail_size <= 0:
        raise ValueError("window_size and tail_size must be positive")
    if any(h <= 0 for h in args.hidden_layer):
        raise ValueError("Hidden layer sizes must be positive")
    if args.save_interval * args.total_saving <= 0:
        raise ValueError("Total training steps must be positive")

def build_training_env(
    topology_file: str,
    node_file: str,
    trace_file: str,
    window_size: int,
    tail_size: int,
    seed: int | None,
) -> HPCsim:
    return HPCsim(
        topology_file=f"data/topology/{topology_file}",
        allocator="best_fit",
        node_file=f"data/topology/{node_file}",
        trace_file=f"data/{trace_file}",
        random_job=False,
        window_size=window_size,
        tail_size=tail_size,
        seed=seed,
    )


def build_model(
    env: HPCsim,
    algorithm: str,
    hidden_layer: list[int],
    activation_fn: str,
    gamma: float,
    seed: int | None,
    buffer_size: int,
    logdir: str,
) -> Any:
    Path(logdir).mkdir(parents=True, exist_ok=True)
    algo_class, policy_kwargs = resolve_algorithm_config(
        algorithm=algorithm,
        hidden_layer=hidden_layer,
        activation_fn=ACTIVATION_MAP[activation_fn],
    )

    model_kwargs: dict[str, Any] = {
        "policy_kwargs": policy_kwargs,
        "gamma": gamma,
        "seed": seed,
        "verbose": 1,
        "tensorboard_log": logdir,
    }

    if "dqn" in algorithm.lower():
        model_kwargs["buffer_size"] = buffer_size

    return algo_class("MultiInputPolicy", env, **model_kwargs)


# ---------------------------------------------------------------------------
# Training and logging
# ---------------------------------------------------------------------------


def train_and_log(
    model: Any,
    treatment_id: str,
    algorithm: str,
    use_masking: bool,
    window_size: int,
    tail_size: int,
    save_interval: int,
    total_saving: int,
    name: str,
    split_id: str,
    trace_file: str,
    topology_file: str,
    node_file: str,
    seed: int | None,
    hyperparams: dict[str, Any],
    logdir: str,
) -> None:
    models_dir = Path(f"trained_model/{name}")
    selector_dir = models_dir / "selector"
    models_dir.mkdir(parents=True, exist_ok=True)
    selector_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = save_interval * total_saving

    if "mask" not in algorithm.lower():
        use_masking = False

    learn_kwargs: dict[str, Any] = {
        "total_timesteps": total_timesteps,
        "tb_log_name": name,
        "callback": SelectorCheckpointCallback(
            save_freq=save_interval,
            save_path=str(selector_dir),
        ),
    }
    if "mask" in algorithm.lower():
        learn_kwargs["use_masking"] = use_masking

    t_start = time.perf_counter()
    model.learn(**learn_kwargs)
    wall_clock_s = time.perf_counter() - t_start

    episodes_completed = int(model._episode_num)

    model_path = str(selector_dir / f"{total_timesteps}.zip")

    run_id = write_manifest_entry(
        treatment_id=treatment_id,
        algorithm=algorithm,
        use_masking=use_masking,
        seed=seed,
        window_size=window_size,
        tail_size=tail_size,
        split_id=split_id,
        model_path=model_path,
        trace_file=trace_file,
        topology_file=topology_file,
        node_file=node_file,
    )

    metadata = build_train_metadata(
        command_args=sys.argv,
        split_id=split_id,
        run_id=run_id,
        treatment_id=treatment_id,
        algorithm=algorithm,
        use_masking=use_masking,
        window_size=window_size,
        tail_size=tail_size,
        seed=seed,
        total_timesteps=total_timesteps,
        save_interval=save_interval,
        total_saving=total_saving,
        model_dir=models_dir,
        selector_dir=selector_dir,
        hyperparams={**hyperparams, "tensorboard_log": logdir},
        wall_clock_s=wall_clock_s,
        episodes_completed=episodes_completed,
    )
    write_json(metadata, models_dir / "train_metadata.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as e:
        print(f"[ERROR] Argument validation failed: {e}")
        sys.exit(1)

    if args.seed is not None:
        set_random_seed(args.seed, using_cuda=True)

    try:
        split_metadata = load_split_metadata(split_id=args.split_id)
        split_id = split_metadata["split_id"]
    except Exception as e:
        print(f"[ERROR] Could not load split metadata: {e}")
        sys.exit(1)

    env = build_training_env(
        topology_file=args.topology_file,
        node_file=args.node_file,
        trace_file=args.trace_file,
        window_size=args.window_size,
        tail_size=args.tail_size,
        seed=args.seed,
    )

    model = build_model(
        env=env,
        algorithm=args.algorithm,
        hidden_layer=args.hidden_layer,
        activation_fn=args.activation_fn,
        gamma=args.gamma,
        seed=args.seed,
        buffer_size=args.buffer_size,
        logdir="logs",
    )

    hyperparams = {
        "window_size": args.window_size,
        "tail_size": args.tail_size,
        "gamma": args.gamma,
        "hidden_layer": args.hidden_layer,
        "activation_fn": args.activation_fn,
        "buffer_size": args.buffer_size,
        "save_interval": args.save_interval,
        "total_saving": args.total_saving,
    }

    if "mask" in args.algorithm.lower():
        use_masking = args.use_masking
    else:
        use_masking = False

    treatment_id = args.algorithm.lower() + "__mask_" + str(use_masking).lower()

    train_and_log(
        model=model,
        treatment_id=treatment_id,
        algorithm=args.algorithm,
        use_masking=use_masking,
        window_size=args.window_size,
        tail_size=args.tail_size,
        save_interval=args.save_interval,
        total_saving=args.total_saving,
        name=args.name,
        split_id=split_id,
        trace_file=args.trace_file,
        topology_file=args.topology_file,
        node_file=args.node_file,
        seed=args.seed,
        hyperparams=hyperparams,
        logdir=f"logs/{args.name}",
    )

    print("[DONE]")

if __name__ == "__main__":
    main()
