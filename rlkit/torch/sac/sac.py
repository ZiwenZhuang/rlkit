from collections import OrderedDict

import numpy as np
import random
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            use_reward_indicator= False, # you can decided whether to use reward filtering mechanism
            filtering_probs= (0.9, 0.5), # This is the two hyper-parameters p_1 and p_2
            use_reward_filter= False,    # available when 'use_reward_indicator' is True,
            reward_filtering_threshold= 2,# the filtering threshold while 'use_reward_filter' is True
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.use_reward_indicator = use_reward_indicator
        self.use_reward_filter = use_reward_filter
        self._filtering_probs = filtering_probs
        self._reward_filtering_threshold = reward_filtering_threshold

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        '''
        Do reward filtering if use choose so.
        '''
        if self.use_reward_indicator:
            achieved_goals = batch['achieved_goals']
            indices = batch['indices']
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions.detach()),
                self.qf2(obs, new_obs_actions.detach()),
            )
            q_new_actions = q_new_actions.detach()
            transition_discarded = []
            # for each transition in R do...
            for i in range(batch["actions"].shape[0]):
                if random.random() < self._filtering_probs[0]:
                    obs_size = obs.shape[1]
                    if random.random() < self._filtering_probs[1]:
                        # change goal as observation
                        obs[i, obs_size // 2:] = obs[i, :obs_size // 2]
                        # recompute the reward (due to the compute_rewards() requirement, I have to expand the array dimension)
                        next_observation = {
                            'achieved_goal': np.expand_dims(obs[i, :obs_size // 2].cpu().numpy(), axis=0),
                            'desired_goal': np.expand_dims(obs[i, obs_size // 2:].cpu().numpy(), axis=0),
                            'image_achieved_goal': np.expand_dims(obs[i, :obs_size // 2].cpu().numpy(), axis=0),
                            'image_desired_goal': np.expand_dims(obs[i, obs_size // 2:].cpu().numpy(), axis=0),
                        }
                        action = actions[i].cpu().numpy()
                        new_reward = self.env.compute_rewards(actions= np.expand_dims(action, axis=0), obs= next_observation)
                        rewards[i] = torch.from_numpy(new_reward)
                    else:
                        # sample from later observations as goal_state
                        choose_ind = np.random.randint((indices.data.cpu().numpy())[i], achieved_goals.shape[0])
                        obs[i, obs_size // 2:] = achieved_goals[choose_ind, :obs_size // 2]
                        # recompute the reward (due to the compute_rewards() requirement, I have to expand the array dimension)
                        next_observation = {
                            'achieved_goal': np.expand_dims(obs[i, :obs_size // 2].cpu().numpy(), axis=0),
                            'desired_goal': np.expand_dims(obs[i, obs_size // 2:].cpu().numpy(), axis=0),
                            'image_achieved_goal': np.expand_dims(obs[i, :obs_size // 2].cpu().numpy(), axis=0),
                            'image_desired_goal': np.expand_dims(obs[i, obs_size // 2:].cpu().numpy(), axis=0),
                        }
                        action = actions[i].cpu().numpy()
                        new_reward = self.env.compute_rewards(actions= np.expand_dims(action, axis=0), obs= next_observation)
                        rewards[i] = torch.from_numpy(new_reward)
                else:
                    # do nothing (allow the reward to be negative)
                    pass
                # the line below is the reward filtering part
                if self.use_reward_filter and rewards[i] == -1 and q_new_actions[i] > self._reward_filtering_threshold:
                    # record the transition to be discarded, not delete now
                    transition_discarded.append(i)
            # trim transitions if needed
            for index in transition_discarded:
                rewards = torch.cat((rewards[:index, :], rewards[index+1:, :]))
                terminals = torch.cat((terminals[:index, :], terminals[index+1:, :]))
                obs = torch.cat((obs[:index, :], obs[index+1:, :]))
                actions = torch.cat((actions[:index, :], actions[index+1:, :]))
                next_obs = torch.cat((next_obs[:index, :], next_obs[index+1:, :]))
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        
        # The 'log_pi' and 'q_new_actions' could be trimed by reward filtering
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )

