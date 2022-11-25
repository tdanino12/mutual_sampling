from policy import DQNTorchPolicy as MyTorchPolicy

from ray.rllib.algorithms.dqn.dqn import *
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

# Create a new Algorithm using the Policy defined above.
class custom_trainer(DQN):
    _allow_unknown_configs = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.td_err = {agent: [] for agent in self.config["multiagent"]["policies"].keys()}
    
    @override(DQN)
    def training_step(self) -> ResultDict:
        """DQN training iteration function.

        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        global rewards
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            new_sample_batch = synchronous_parallel_sample(
                worker_set=self.workers, concat=True
            )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED if self._by_agent_steps else NUM_ENV_STEPS_SAMPLED
        ]

        if cur_ts > self.config["num_steps_sampled_before_learning_starts"]:
            for _ in range(sample_and_train_weight):
                # Sample training batch (MultiAgentBatch) from replay buffer.
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config["train_batch_size"],
                    count_by_agent_steps=self._by_agent_steps,
                )
                
                ####### Mutual sampling #######
                if self.config["mutual_sampling"]:
                    sample_weight, performers_sorted = {}, []

                    for agent in new_sample_batch.policy_batches:
                        # get policy object for specific agent
                        agent_policy = self.workers.local_worker().policy_map[agent]

                        # get td_errors from last training batch
                        batch_agent = train_batch[agent]
                        td_err_tensors = agent_policy.compute_td_error(batch_agent["obs"], batch_agent["actions"], batch_agent["rewards"],
                                      batch_agent["new_obs"], batch_agent["dones"], batch_agent["weights"]).detach().cpu()


                        # Add curr experiences to saved td-errors
                        td_abs_list = list(map(abs, td_err_tensors.tolist()))
                        self.td_err[agent].append(sum(td_abs_list))

                        # sort agents according to their td error
                        memory = self.config["majority_memory"]
                        # give weight to each agent according to his td-err
                        sample_weight[agent] = sum(self.td_err[agent][-memory:]) / len(self.td_err[agent][-memory:])
                        performers_sorted = [k for k, v in sorted(sample_weight.items(), key=lambda item: item[1])]

                    mini_sample_size = int(self.config["mutual_batch_addition"]/self.config["mutual_leaders"])

                    leaders_batch = SampleBatch.concat_samples([self.local_replay_buffer.sample(mini_sample_size,agent) \
                                                               for agent in performers_sorted[-self.config["mutual_leaders"]:]])

                    for agent in new_sample_batch.policy_batches:
                        train_batch[agent].concat_samples([leaders_batch])

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                # for policy_id, sample_batch in train_batch.policy_batches.items():
                #     print(len(sample_batch["obs"]))
                #     print(sample_batch.count)

                # Learn on training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config["target_network_update_freq"]:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda p, pid: pid in to_update and p.update_target()
                    )
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers.
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results


# Deprecated: Use ray.rllib.algorithms.dqn.DQNConfig instead!
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(DQNConfig().to_dict())

    @Deprecated(
        old="ray.rllib.algorithms.dqn.dqn.DEFAULT_CONFIG",
        new="ray.rllib.algorithms.dqn.dqn.DQNConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()
