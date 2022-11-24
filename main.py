import ray
import torch
import numpy as np
from uuid import uuid4
import datetime
from ray import air, tune
from ray.tune.registry import register_env
from social_dilemmas.envs.env_creator import get_env_creator
from ray.air.callbacks.wandb import WandbLoggerCallback
import argparse
from env import *
from config import *
from trainer import *


def set_seeds():
    SEED = int(os.environ.get("SEED", 42))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    seeding.np_random(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    return SEED

def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="pursuit",
        help="Determines the training environment",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPU cores",
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        default="custom_policy",
        help="Run group to use for logging",
    )
    parser.add_argument(
        "--experiment_project",
        type=str,
        default="majority_dqn2",
        help="wandb project to use for logging",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}",
        help="Run name to use for logging",
    )
    parser.add_argument(
        "--ray_local_mode",
        action="store_true",
        help="If enabled, init ray in local mode.",
    )

    parser.add_argument(
        "--majority",
        default="true",
        help="If enabled, use majority dqn.",
    )

    parser.add_argument(
        "--double_q",
        default="true",
        help="If enabled, use majority dqn.",
    )

    parser.add_argument(
        "--majority_prob",
        type=int,
        default=9,
        help="a number between 0 to 10 that determined the probability that majority will be used",
    )

    parser.add_argument(
        "--majority_full",
        default="false",
        help="if set to true, q values are not summed",
    )

    # arguments for battle environment
    parser.add_argument("--env_battle_map_size", type=int, required=False, help="Override map size for battle?")

    args, remaining_cli = parser.parse_known_args(arg_string)
    return args, remaining_cli


def main(args, num_cpus, group: str = "suPER",
         name: str = f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}", ray_local_mode: bool = False):
    SEED = set_seeds()
    if args.env == "pursuit":
        trainer, config, stop = config_pursuit_dqn(args)
        register_env("pursuit_v4", lambda config: manual_wrap(env_creator_pursuit(config), SEED))
    elif args.env == "battle":
        register_env("battle_v4", lambda config: manual_wrap(env_creator_battle_v4(config), SEED))
        config, stop = config_battlev4_dqn(args)
    elif args.env == "adversarial_pursuit":
        register_env("adversarial_pursuit", lambda config: manual_wrap(env_creator_adversarial_pursuit(config), SEED))
        config, stop = config_adversarial_pursuit_dqn(args)
    elif args.env == "simple_tag":
        register_env("simple_tag_v2", lambda config: manual_wrap(env_creator_simpletag(config), SEED))
        config, stop = config_simple_tag(args)
    elif args.env == "simple_spread":
        register_env("simple_spread_v2", lambda config: manual_wrap(env_creator_spread(config), SEED))
        config, stop = config_simple_spread(args)
    elif args.env == "cleanup":
        config, stop = config_cleanup_dqn(args)
        env_creator = get_env_creator("cleanup", config["env_config"]["num_agents"], config["env_config"]["use_collective_reward"], config["env_config"]["num_switches"])
        register_env(config["env"], env_creator)
    elif args.env == "harvest":
        config, stop = config_harvest_dqn(args)
        env_creator = get_env_creator("harvest", config["env_config"]["num_agents"], config["env_config"]["use_collective_reward"], config["env_config"]["num_switches"])
        register_env(config["env"], env_creator)

    else:
        raise NotImplementedError(f"Environment {args.env} not implemented.")

    if args.double_q == "true":
        config["double_q"] = True
    if args.majority == "true":
        config["majority"] = True
    if args.majority_full == "true":
        config["pure_majority"] = "true"


    ray.init(num_cpus=num_cpus, local_mode=args.ray_local_mode, include_dashboard=False)
    curr_folder = os.path.dirname(os.path.realpath(__file__))
    local_dir = curr_folder + "/ray_results/" + uuid4().hex + "/"


    # Set up Weights And Biases logging if API key is set in environment variable.
    if "WANDB_API_KEY" in os.environ:
        callbacks = [
            WandbLoggerCallback(
                project=args.experiment_project,
                api_key=os.environ["WANDB_API_KEY"],
                log_config=True,
                resume=False,
                group=args.env,
                name=name,
                entity="harvardparkesateams",
                id=f"{args.experiment_name}_{os.environ.get('SEED', 42)}_{datetime.datetime.now().timestamp()}_{os.getpid()}",
            )
        ]
    else:
        callbacks = []
        # raise ValueError("No wandb API key specified, aborting.")

    tuner = tune.Tuner(custom_trainer, param_space=config,
                       run_config=air.RunConfig(stop=stop, callbacks=callbacks, verbose=1, local_dir=local_dir))
    tuner.fit()
    ray.shutdown()


if __name__ == "__main__":

    args, remaining_cli = parse_args()
    for a in remaining_cli:
        print(f"WARNING! Ignoring unknown argument {a}.")

    main(args, args.num_cpus, group=args.experiment_group, name=args.experiment_name, ray_local_mode=args.ray_local_mode)