from model import PursuitModel, BattleModel, AdversarialPursuitModel, Simple_tag_model
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.dqn import DQNConfig
from social_dilemmas.envs.env_creator import get_env_creator
from copy import deepcopy
import numpy as np
import torch
import os
import random
from env import *
from gym.utils import seeding

from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
_PURSUIT_N_TIMESTEPS = 1300000
_BATTLE_N_TIMESTEPS = 600000
_ADVPURSUIT_N_TIMESTEPS = 700000
_EVAL_RESOLUTION = 200


def set_seeds():
    SEED = int(os.environ.get("SEED", 42))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    seeding.np_random(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    return SEED


def create_model_config(agent_id, model):
    if model == "Simple_tag_model":
        model_config = {
                "id": agent_id,
            }
    elif model == "harvest":
        model_config = {
                "id": agent_id,
            }
    elif model == "cleanup":
        model_config = {
                "id": agent_id,
            }
    else:
        model_config = {
                "model": {
                    "custom_model": model,
                },
                "gamma": 0.99,
                "id": agent_id,
            }

    return model_config


def config_pursuit_dqn(args):
    set_seeds()
    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("PursuitModel", PursuitModel)

    model_config = {
            "model": {
                "custom_model": "PursuitModel",
            },
            "gamma": 0.99,
        }
    env_config = {"num_agents": 8, "n_evaders": 30, "shared_reward": False}
    '''
    # Overwrite env_config variables using CLI args, if given.
    if args.env_pursuit_num_agents is not None:
        env_config["num_agents"] = args.env_pursuit_num_agents
    if args.env_pursuit_n_evaders is not None:
        env_config["n_evaders"] = args.env_pursuit_n_evaders
    if args.env_pursuit_shared_reward is not None:
        env_config["shared_reward"] = args.env_pursuit_shared_reward
    '''

    policies = {f"pursuer_{i}": PolicySpec(None, None, None, create_model_config(f"pursuer_{i}", "PursuitModel"))
                for i in range(env_config["num_agents"])}

    config = deepcopy(DQNConfig().to_dict())

    config["env"] = "pursuit_v4"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    #config["log_level"] = "INFO"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06
    config["train_batch_size"] = env_config["num_agents"] * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.00016
    config["horizon"] = 500
    config["dueling"] = True
    config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    config["majority"] = False
    config["does_majority"] = args.majority_prob/10  # the probability to choose majority.
    config["majority_agents"] = []  # which agent allowed to use majority, if empty, all are allowed.
    config["majority_weight"] = 0.8  # weight that determines the q estimation proportion.
    config["action_space_size"] = 5
    config["majority_leaders"] = 4
    config["majority_memory"] = 3500
    config["pure_majority"] = "false"

    config["mutual_batch_addition"] = 18
    config["mutual_leaders"] = 3
    config["mutual_sampling"] = True

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 20
    config["evaluation_duration"] = 90
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20
    config["double_q"] = False
    #config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}

    config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    return "DQN", config, stop


def config_simple_spread(args):
    set_seeds()
    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("Simple_tag_model", Simple_tag_model)

    model_config = {
            "model": {
                "custom_model": "Simple_tag_model",
            },
            "gamma": 0.99,
        }
    env_config = {"num_agents": 3, "max_cycles": 25}

    policies_list = ["agent_{}".format(i) for i in range(env_config["num_agents"])]

    policies = {i: PolicySpec(None, None, None, create_model_config(i, "Simple_tag_model"))
                for i in policies_list}

    config = deepcopy(DQNConfig().to_dict())

    config["env"] = "simple_spread_v2"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    #config["log_level"] = "INFO"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06
    config["train_batch_size"] = len(policies_list) * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.00016
    config["horizon"] = env_config["max_cycles"]
    config["dueling"] = True
    config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    config["majority"] = False
    config["does_majority"] = args.majority_prob/10  # the probability to choose majority.
    config["majority_agents"] = ["agent_0", "agent_1", "agent_2"]  # which agent allowed to use majority, if empty, all are allowed.
    config["majority_weight"] = 0.3  # weight that determines the q estimation proportion.
    config["action_space_size"] = 5
    config["pure_majority"] = "false"
    config["majority_leaders"] = 3
    config["majority_memory"] = 3500

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 20
    config["evaluation_duration"] = 90
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20
    config["double_q"] = False
    #config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}
    config["model"] = {"fcnet_hiddens": [64, 64]}
    config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    return config, stop


def config_simple_tag(args):
    set_seeds()
    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("Simple_tag_model", Simple_tag_model)

    model_config = {
            "model": {
                "custom_model": "Simple_tag_model",
            },
            "gamma": 0.99,
        }
    env_config = {"good_agents": 1, "adversaries_agents": 3, "num_obstacles": 3, "max_cycles": 35}

    policies_good_agents = ["agent_{}".format(i) for i in range(env_config["good_agents"])]
    policies_adversaries_agents = ["adversary_{}".format(i) for i in range(env_config["adversaries_agents"])]
    policies_list = policies_good_agents + policies_adversaries_agents

    policies = {i: PolicySpec(None, None, None, create_model_config(i, "Simple_tag_model"))
                for i in policies_list}

    config = deepcopy(DQNConfig().to_dict())

    config["env"] = "simple_tag_v2"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    #config["log_level"] = "INFO"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 1e-06
    config["train_batch_size"] = len(policies_list) * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.00016
    config["horizon"] = env_config["max_cycles"]
    config["dueling"] = True
    config["target_network_update_freq"] = 1000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False

    config["majority"] = False
    config["does_majority"] = args.majority_prob/10  # the probability to choose majority.
    config["majority_agents"] = ["adversary_0", "adversary_1", "adversary_2"]  # which agent allowed to use majority, if empty, all are allowed.
    config["majority_weight"] = 0.3  # weight that determines the q estimation proportion.
    config["action_space_size"] = 5
    config["pure_majority"] = "false"
    config["majority_leaders"] = 3
    config["majority_memory"] = 3500

    config["min_sample_timesteps_per_iteration"] = _PURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 20
    config["evaluation_duration"] = 90
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20
    config["double_q"] = False
    #config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}
    config["model"] = {"fcnet_hiddens": [64, 64]}
    config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    stop = {"timesteps_total": _PURSUIT_N_TIMESTEPS}

    return config, stop


def config_battlev4_dqn(args):
    set_seeds()

    env_config = {"map_size": 18}

    # Register custom model for Pursuit environment.
    ModelCatalog.register_custom_model("BattleModel", BattleModel)

    model_config = {
            "model": {
                "custom_model": "BattleModel",
            },
            "gamma": 0.99,
    }

    # Overwrite env_config variables using CLI args, if given.
    if args.env_battle_map_size is not None:
        env_config["map_size"] = args.env_battle_map_size

    env = env_creator_battle_v4(env_config)
    num_battle_agents = len(env.agents)
    policies_blue_battle = ["blue_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_red_battle = ["red_{}".format(i) for i in range(int(num_battle_agents / 2))]
    policies_list = policies_blue_battle + policies_red_battle
    policies = {i: PolicySpec(None, None, None, create_model_config(i, "BattleModel")) for i in policies_list}
    config = deepcopy(DQNConfig().to_dict())
    config["env"] = "battle_v4"
    config["env_config"] = env_config
    config["store_buffer_in_checkpoints"] = False
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 90000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["rollout_fragment_length"] = 5
    config["prioritized_replay_alpha"] = 0.6
    config["prioritized_replay_beta"] = 0.4
    config["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = num_battle_agents * 32  # each agent's sample batch is with size 32
    config["replay_buffer_config"]["worker_side_prioritization"] = True
    config["lr"] = 1e-4
    config["horizon"] = 1000
    config["dueling"] = True
    config["target_network_update_freq"] = 1200
    config["no_done_at_end"] = False
    config["num_steps_sampled_before_learning_starts"] = 1000
    config["multiagent"] = {"policies": policies}
    config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id
    config["min_sample_timesteps_per_iteration"] = _BATTLE_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 20
    config["evaluation_duration"] = 90
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20
    config["action_space_size"] = 21
    config["majority"] = False
    config["does_majority"] = args.majority_prob / 10  # the probability to choose majority.
    config["majority_weight"] = 0.3  # weight that determines the q estimation proportion.
    config["majority_agents"] = ["blue_0", "blue_1", "blue_2",
                                 "blue_3", "blue_4", "blue_5"]  # which agent allowed to use majority, if empty, all are allowed.
    config["pure_majority"] = "false"
    config["majority_leaders"] = 3
    config["majority_memory"] = 3500

    stop = {"timesteps_total": _BATTLE_N_TIMESTEPS}

    return config, stop


def config_adversarial_pursuit_dqn(args):
    set_seeds()
    # num of predators and prey is determined according to map size,
    # and is not given as input to the environment.
    num_predators = 4
    num_preys = 8

    # map size sets dimensions of the (square) map.
    env_config = {"map_size": 18}

    ModelCatalog.register_custom_model("AdversarialPursuitModel", AdversarialPursuitModel)

    model_config = {
            "model": {
                "custom_model": "AdversarialPursuitModel",
            },
            "gamma": 0.99,
            "id": 2,
        }

    policies_predator_battle = ["predator_{}".format(i) for i in range(num_predators)]
    policies_prey_battle = ["prey_{}".format(i) for i in range(num_preys)]
    policies_list = policies_prey_battle + policies_predator_battle
    policies = {i: PolicySpec(None, None, None, create_model_config(i, "AdversarialPursuitModel")) for i in policies_list}

    config = deepcopy(DQNConfig().to_dict())
    config["env"] = "adversarial_pursuit"
    config["env_config"] = env_config
    config["store_buffer_in_checkpoints"] = False
    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 90000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["rollout_fragment_length"] = 5
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = (num_predators + num_preys) * 32  # each agent's sample batch is with size 32
    config["replay_buffer_config"]["worker_side_prioritization"] = True
    config["lr"] = 1e-4
    config["horizon"] = 800
    config["dueling"] = True
    config["target_network_update_freq"] = 1200
    config["no_done_at_end"] = False
    config["multiagent"] = {"policies": policies}
    config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id

    config["min_sample_timesteps_per_iteration"] = _ADVPURSUIT_N_TIMESTEPS / _EVAL_RESOLUTION
    config["min_time_s_per_iteration"] = 0
    config["evaluation_interval"] = 20
    config["evaluation_duration"] = 90
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 20
    config["num_steps_sampled_before_learning_starts"] = 1000

    config["action_space_size"] = 13
    config["majority"] = False
    config["does_majority"] = args.majority_prob / 10  # the probability to choose majority.
    config["majority_weight"] = 0.2  # weight that determines the q estimation proportion.
    config["majority_agents"] = ["prey_0", "prey_1", "prey_2", "prey_3",
                                 "prey_4", "prey_5", "prey_6"]  # which agent allowed to use majority, if empty, all are allowed.
    config["pure_majority"] = "false"
    config["majority_leaders"] = 3
    config["majority_memory"] = 3500

    stop = {"timesteps_total": _ADVPURSUIT_N_TIMESTEPS}

    return config, stop


def config_cleanup_dqn(args):
    SEED = set_seeds()

    # set environment parameters
    config = deepcopy(DQNConfig().to_dict())

    env_config = {"env_name": "cleanup", "num_agents": 5, "use_collective_reward": False, "num_switches": 6}
    '''
    # Overwrite env_config variables using CLI args, if given.
    if args.env_cleanup_num_agents is not None:
        env_config["num_agents"] = args.env_cleanup_num_agents
    if args.env_cleanup_collective_reward is not None:
        env_config["use_collective_reward"] = args.env_cleanup_collective_reward
    if args.env_cleanup_num_switches is not None:
        env_config["num_switches"] = args.env_cleanup_num_switches
    '''

    env_creator = get_env_creator("cleanup", env_config["num_agents"], env_config["use_collective_reward"], env_config["num_switches"])
    single_env = env_creator(env_config["num_agents"])

    obs_space = single_env.observation_space
    act_space = single_env.action_space
    policies = {f"agent-{i}": PolicySpec(None, obs_space, act_space, create_model_config(f"agent_{i}", "cleanup")) for i in range(env_config["num_agents"])}

    config["env"] = "cleanup_env"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = env_config["num_agents"] * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.0001
    config["horizon"] = 1000
    config["dueling"] = True
    config["target_network_update_freq"] = 2000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False
    config["model"] = {"fcnet_hiddens": [512, 512], "conv_filters": [[16, [2, 2], 2], [32, [2, 2], 2], [512, [2, 2], 1]]}
    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}

    config["min_sample_timesteps_per_iteration"] = 4
    config["min_time_s_per_iteration"] = 300
    config["min_sample_timesteps_per_iteration"] = 4
    config["evaluation_interval"] = 20
    config["evaluation_duration"] = 90
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 5
    config["seed"] = SEED
    config["action_space_size"] = 9
    config["majority"] = False
    config["does_majority"] = args.majority_prob / 10  # the probability to choose majority.
    config["majority_weight"] = 0.3  # weight that determines the q estimation proportion.
    config["majority_agents"] = []  # which agent allowed to use majority, if empty, all are allowed.
    config["pure_majority"] = "true"
    config["majority_leaders"] = 4
    config["majority_memory"] = 3500

    stop = {"timesteps_total": 1000000}

    return config, stop


def config_harvest_dqn(args):
    SEED = set_seeds()

    # set environment parameters
    config = deepcopy(DQNConfig().to_dict())

    env_config = {"env_name": "harvest", "num_agents": 5, "use_collective_reward": False, "num_switches": 6}
    '''
    # Overwrite env_config variables using CLI args, if given.
    if args.env_harvest_num_agents is not None:
        env_config["num_agents"] = args.env_harvest_num_agents
    if args.env_harvest_collective_reward is not None:
        env_config["use_collective_reward"] = args.env_harvest_collective_reward
    if args.env_harvest_num_switches is not None:
        env_config["num_switches"] = args.env_harvest_num_switches
    '''

    env_creator = get_env_creator("harvest", env_config["num_agents"], env_config["use_collective_reward"], env_config["num_switches"])
    single_env = env_creator(env_config["num_agents"])

    obs_space = single_env.observation_space
    act_space = single_env.action_space
    policies = {f"agent-{i}": PolicySpec(None, obs_space, act_space, create_model_config(f"agent_{i}", "harvest")) for i in range(env_config["num_agents"])}

    config["env"] = "harvest_env"

    config["env_config"] = env_config

    config["num_gpus"] = 0
    config["framework"] = "torch"
    # config["log_level"] = "DEBUG"
    config["replay_buffer_config"]["capacity"] = 120000
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 0.1,
        "final_epsilon": 0.001,
        "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    }
    config["replay_buffer_config"]["type"] = "MultiAgentPrioritizedReplayBuffer"
    config["replay_buffer_config"]["prioritized_replay_alpha"] = 0.6
    config["replay_buffer_config"]["prioritized_replay_beta"] = 0.4
    config["replay_buffer_config"]["prioritized_replay_eps"] = 0.00001
    config["train_batch_size"] = env_config["num_agents"] * 32  # each agent's sample batch is with size 32
    config["lr"] = 0.0001
    config["horizon"] = 1000
    config["dueling"] = True
    config["target_network_update_freq"] = 2000
    config["rollout_fragment_length"] = 4
    config["no_done_at_end"] = False
    config["model"] = {"fcnet_hiddens": [512, 512], "conv_filters": [[16, [2, 2], 2], [32, [2, 2], 2], [512, [2, 2], 1]]}
    config["multiagent"] = {"policies": policies, "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id)}
    config["seed"] = SEED
    config["min_sample_timesteps_per_iteration"] = 4
    config["min_time_s_per_iteration"] = 300
    config["min_sample_timesteps_per_iteration"] = 4
    config["evaluation_interval"] = 12
    config["evaluation_duration"] = 40
    config["evaluation_duration_unit"] = "episodes"
    config["metrics_num_episodes_for_smoothing"] = 5

    config["action_space_size"] = 8
    config["majority"] = False
    config["does_majority"] = args.majority_prob / 10  # the probability to choose majority.
    config["majority_weight"] = 0.2  # weight that determines the q estimation proportion.
    config["majority_agents"] = []  # which agent allowed to use majority, if empty, all are allowed.
    config["pure_majority"] = "true"
    config["majority_leaders"] = 3
    config["majority_memory"] = 3500

    config["model"] = {"fcnet_hiddens": [512, 512],
                       "conv_filters": [[16, [2, 2], 2], [32, [2, 2], 2], [512, [2, 2], 1]]}

    stop = {"timesteps_total": 1000000}

    return config, stop