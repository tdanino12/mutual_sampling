import os
import pickle
import numpy as np
import ray
import supersuit
from gym import spaces
from pettingzoo.sisl import pursuit_v4
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.magent import adversarial_pursuit_v4, battle_v4
from pettingzoo.mpe import simple_tag_v2, simple_spread_v2
from supersuit import black_death_v3, pad_action_space_v0, pad_observations_v0


class manual_wrap(ParallelPettingZooEnv):
    """This class is used only for the Pursuit environment. It inherits
    the ParallelPettingZooEnv wrapper, and overwrites the reset logic. In the current
    version of Pursuit, seeding the environment cannot be done via the config file, but rather
    through the reset command. RLlib don't support seeding via reset yet, hence
    we provide this customized wrapper class.
    """

    def __init__(self, env, seed):
        super().__init__(env)
        env.reset(seed=seed)

    def reset(self):
        return self.par_env.reset()

    def step(self, action_dict):
        obss, rews, dones, infos = super().step(action_dict)
        return obss, rews, dones, infos

    def seed(self, seed=None):
        pass


def env_creator_pursuit(env_config):
    """This function creates a parallel Pursuit environment object.
    We use default configurations provided by petting zoo.
    """
    env = pursuit_v4.parallel_env(
        max_cycles=500,
        x_size=16,
        y_size=16,
        shared_reward=env_config["shared_reward"],
        n_evaders=env_config["n_evaders"],
        n_pursuers=env_config["num_agents"],
        obs_range=7,
        n_catch=2,
        freeze_evaders=False,
        tag_reward=0.01,
        catch_reward=5.0,
        urgency_reward=-0.1,
        surround=True,
        constraint_window=1.0,
    )

    return env


def env_creator_battle_v4(env_config):
    """This function creates a parallel Battled environment object.
    We use default configurations provided by petting zoo.
    """
    env = battle_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)
    return env


def env_creator_adversarial_pursuit(env_config):
    env = adversarial_pursuit_v4.parallel_env(map_size=env_config["map_size"], minimap_mode=False, tag_penalty=-0.2, max_cycles=500, extra_features=False)

    # Predator and prey have different action space and state sizes,
    # since Rlib requires all agents to have identical state and action size,
    # padding is required.
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    return env


def env_creator_simpletag(env_config):
    env = simple_tag_v2.parallel_env(num_good=env_config["good_agents"],
                            num_adversaries=env_config["adversaries_agents"],
                            num_obstacles=env_config["num_obstacles"],
                            max_cycles=env_config["max_cycles"], continuous_actions=False)
    env = pad_observations_v0(env)
    return env


def env_creator_spread(env_config):
    env = simple_spread_v2.parallel_env(N=env_config["num_agents"],
                               local_ratio=0.5,
                               max_cycles=env_config["max_cycles"],
                               continuous_actions=False)
    return env