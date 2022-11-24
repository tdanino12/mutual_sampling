from ray.rllib.algorithms.dqn.dqn_torch_policy import *
import numpy as np
from random import random
policies = {}
models = {}
rewards = {}
td_errors = {}


def build_q_losses(policy: Policy, model, _, train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for DQNTorchPolicy.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        train_batch: The training data.

    Returns:
        TensorType: A single loss tensor.
    """
    global rewards, policies, models, td_errors
    agent_index = policy.config["id"]  # int(train_batch["agent_index"][0].item())

    config = policy.config

    #########################
    # if you are on the team that use majority td, or all agents use majority, save your policy/model.
    if config["majority"] and \
            (len(config["majority_agents"]) == 0 or agent_index in config["majority_agents"]):
        policies[agent_index] = policy
        models[agent_index] = model
    #########################

    # Q-network evaluation.
    q_t, q_logits_t, q_probs_t, _ = compute_q_values(
        policy,
        model,
        {"obs": train_batch[SampleBatch.CUR_OBS]},
        explore=False,
        is_training=True,
    )

    # Target Q-network evaluation.
    q_tp1, q_logits_tp1, q_probs_tp1, _ = compute_q_values(
        policy,
        policy.target_models[model],
        {"obs": train_batch[SampleBatch.NEXT_OBS]},
        explore=False,
        is_training=True,
    )

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(
        train_batch[SampleBatch.ACTIONS].long(), policy.action_space.n
    )
    q_t_selected = torch.sum(
        torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
        * one_hot_selection,
        1,
    )
    q_logits_t_selected = torch.sum(
        q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
    )

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        (
            q_tp1_using_online_net,
            q_logits_tp1_using_online_net,
            q_dist_tp1_using_online_net,
            _,
        ) = compute_q_values(
            policy,
            model,
            {"obs": train_batch[SampleBatch.NEXT_OBS]},
            explore=False,
            is_training=True,
        )

        #########################
        #  if majority = True, and you are on the team that use majority td, follow the majority logic.
        if config["majority"] and random() > config["does_majority"] and\
                (len(config["majority_agents"]) == 0 or agent_index in config["majority_agents"]):

            # sort agents accotding to their avg td_error
            performers = {}
            for p in td_errors:
                if len(td_errors[p]) > 0:
                    memory = min(config["majority_memory"], len(td_errors[p]))
                    performers[p] = sum(td_errors[p][-memory:])/len(td_errors[p][-memory:])
            performers_sorted = [k for k, v in sorted(performers.items(), key=lambda item: item[1])]

            a_size = config["action_space_size"]
            q_tp1_total = [[0 for j in range(a_size)] for i in range(32)]  # used to determine optimal action
            q_tp1_sum = [[0 for j in range(a_size)] for i in range(32)]  # used to estimate q value
            q_tp1_total_argmax = [[0 for j in range(a_size)] for i in range(32)]
            count_leaders = 0
            # iterate through best performing agents and ask for their q value
            for p in performers_sorted[0:config["majority_leaders"]]:
                if p != agent_index:
                    count_leaders += 1

                    # target Q-network estimation (for q value estimation)
                    q_tp1_other, q_logits_tp1_other, q_probs_tp1_other, _ = compute_q_values(
                        policies[p],
                        policies[p].target_models[models[p]],
                        {"obs": train_batch[SampleBatch.NEXT_OBS]},
                        explore=False,
                        is_training=True,
                    )

                    # online network evaluation (for action selection).
                    q_tp1_online_other, q_logits_tp1_online_other, q_probs_online_tp1_other, _ = compute_q_values(
                        policies[p],
                        models[p],
                        {"obs": train_batch[SampleBatch.NEXT_OBS]},
                        explore=False,
                        is_training=True,
                    )
                    numpy_arr = q_tp1_online_other.detach().numpy()  # used to evaluate optimal action
                    numpy_arr_sum = q_tp1_other.detach().numpy()  # used to estimate q value

                    q_tp1_total = np.add(q_tp1_total, numpy_arr)
                    q_tp1_sum = np.add(q_tp1_sum, numpy_arr_sum)

                    if(config["pure_majority"]) == "true":
                        for count, iter in enumerate(q_tp1_total):
                            q_tp1_total_argmax[count][iter.argmax()] += 1

            if(config["pure_majority"]) == "false":
                q_tp1_total = (1-config["majority_weight"])*np.array(q_tp1_total)
                q_original = config["majority_weight"]*np.array(q_tp1_using_online_net.tolist())
                q_tp1_total = np.add(q_tp1_total, q_original)
                # take the action with the highest qumaluative q value
                q_tp1_best_using_online_net = torch.argmax(torch.tensor(q_tp1_total), 1)
            else:
                q_tp1_best_using_online_net = torch.argmax(torch.tensor(q_tp1_total_argmax), 1)

            # edit q_tp1 and change it to the mean q value of all selected estimators,
            numpy_arr_sum = q_tp1.detach().numpy()  # take original q_tp1
            q_tp1_sum = np.add(q_tp1_sum, numpy_arr_sum)  # add other agents q_tp1
            q_tp1_sum = np.divide(q_tp1_sum, count_leaders+1)  # avg values
            q_tp1 = torch.tensor(q_tp1_sum).float()  # convert back to tensors
        #########################
        else:
            q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)

        q_tp1_best_one_hot_selection = F.one_hot(
            q_tp1_best_using_online_net, policy.action_space.n
        )

        q_tp1_best = torch.sum(
            torch.where(
                q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
            )
            * q_tp1_best_one_hot_selection,
            1,
        )

        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
        )

    else:
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), policy.action_space.n
        )
        q_tp1_best = torch.sum(
            torch.where(
                q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
            )
            * q_tp1_best_one_hot_selection,
            1,
        )
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
        )

    q_loss = QLoss(
        q_t_selected,
        q_logits_t_selected,
        q_tp1_best,
        q_probs_tp1_best,
        train_batch[PRIO_WEIGHTS],
        train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.DONES].float(),
        config["gamma"],
        config["n_step"],
        config["num_atoms"],
        config["v_min"],
        config["v_max"],
    )

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["td_error"] = q_loss.td_error
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["q_loss"] = q_loss

    return q_loss.loss


def policy_post_fn(policy: Policy, batch: SampleBatch, other_agent=None, episode=None):
    global rewards, policies, models, td_errors
    config = policy.config
    agent_index = policy.config["id"]  # int(train_batch["agent_index"][0].item())
    batch = postprocess_nstep_and_prio(policy, batch, other_agent, episode)
    td_tensor_list = policy.compute_td_error(batch["obs"], batch["actions"], batch["rewards"],
                                  batch["new_obs"], batch["dones"], batch["weights"]).detach().cpu()
    td_abs_list = list(map(abs, td_tensor_list.tolist()))
    td_sum = sum(td_abs_list)
    reward_sum = sum(batch["rewards"].tolist())
    #########################
    if len(config["majority_agents"]) == 0 or agent_index in config["majority_agents"]:
        if agent_index not in rewards:
            rewards[agent_index] = []
            td_errors[agent_index] = []
        else:
            td_errors[agent_index].append(td_sum)
            rewards[agent_index].append(reward_sum)
    #########################
    return batch


DQNTorchPolicy = build_policy_class(
    name="DQNTorchPolicy",
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DEFAULT_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=policy_post_fn,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)
