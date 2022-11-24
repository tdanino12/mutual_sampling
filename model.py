import os
import random
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

SEED = int(os.environ.get("SEED", 42))
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
seeding.np_random(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


class PursuitModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(1024, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class Simple_tag_model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Flatten(),
            (nn.Linear(26, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"])
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class AdversarialPursuitModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(5, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class BattleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(5, 32, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2]),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(6400, 64)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()
