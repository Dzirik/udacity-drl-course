"""
Agents
"""
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from numpy import argmax, ndarray, dtype
from numpy.random import random
from torch import optim
from torch.nn.utils import clip_grad_norm_  # type:ignore

from src.data.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.utils.date_time_functions import convert_datetime_to_string_date


# pylint: disable = no-member
class BaseAgent(ABC):
    """
    Base agent to set up interface.
    """

    def __init__(self, env: Any, replay_buffer: Any, gamma: float) -> None:
        self._env = env
        self._memory = replay_buffer

        self._experience: List[Any]  # variable for collecting one-step experience

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._gamma = gamma  # 0.99

        self._q_network_local: Any

    @abstractmethod
    def _get_greedy_action(self, state: Any) -> int:
        """
        Gets the greedy action.
        :param state: Any.
        :return: int. Greedy action.
        """

    @staticmethod
    def get_device_type() -> str:
        """
        Gets the device used.
        :return: str. Type of the device.
        """
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def act(self, state: Any, eps: float = 0.) -> int:
        """
        Selects an action based on current state.
        NOTE: If eps = 0., then always greedy action is taken.
        :param state: Any.
        :param eps: float. Epsilon for greedy choice.
            - P(random_action) = epsilon.
            - P(greedy_action) = 1 - epsilon
        :return: int. Action taken.
        """
        if random() > eps:
            # greedy action
            action = self._get_greedy_action(state)
        else:
            # random action
            action = self._env.action_space.sample()
        self._experience = [state, action]
        return action

    def step(self, action: int) -> Tuple[ndarray[Any, dtype[Any]], float, bool]:
        """
        Takes an action, collects the experience from the environment and stores it into the memory.
        :param action: int. Action taken.
        :return: Tuple[ndarray[Any, dtype[Any]], float, bool]. (next_state, reward, done).
        """
        next_state, reward, done, _, _ = self._env.step(action)
        self._experience = self._experience + [reward, next_state, done]
        self._memory.add(*self._experience)

        return next_state, reward, done

    @abstractmethod
    def learn(self) -> None:
        """
        Learns from the experience collected.
        """

    def save_model(self) -> str:
        """
        Saves the model.
        :return: str. Name of the file name.
        """
        file_name = f"{convert_datetime_to_string_date(datetime.now())}_model_checkpoint.pth"
        torch.save(self._q_network_local.state_dict(), file_name)

        return file_name


# pylint: disable=too-many-instance-attributes
class DQNAgent(BaseAgent):
    """
    Class for double q network agent.
    """

    def __init__(self, env: Any, actions_dim: int, memory_size: int, batch_size: int, q_network_class: Any, \
                 gamma: float) -> None:
        replay_buffer = ReplayBuffer(
            state_dim=env.observation_space.shape[0],
            actions_dim=actions_dim,
            buffer_size=memory_size,
            batch_size=batch_size,
            n_actions=env.action_space.n
        )
        BaseAgent.__init__(self, env=env, replay_buffer=replay_buffer, gamma=gamma)

        self._q_network_local = q_network_class(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            seed=988
        )
        self._q_network_target = q_network_class(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            seed=988
        )
        self._optimizer = optim.Adam(self._q_network_local.parameters(), lr=5e-4)

        self._steps = 0
        self._batch_size = batch_size

        self._update_every_steps = 2
        self._hard_update_every_steps = 8
        self._tau = 0.001

    def set_optimizing_parameters(self, update_every_steps: int, hard_update_every_steps: int, tau: float) -> None:
        """
        Sets neural network training parameters.
        :param update_every_steps: int. After how many steps the local neural network should be updated.
        :param hard_update_every_steps: int. After how many steps the targe neural network should be updated.
        :param tau: float. Hard update parameter.
        """
        self._update_every_steps = update_every_steps
        self._hard_update_every_steps = hard_update_every_steps
        self._tau = tau

    def _get_greedy_action(self, state: Any) -> int:
        """
        Gets the greedy action.
        :param state: Any.
        :return: int. Greedy action.
        """
        net_state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._q_network_local.eval()
        with torch.no_grad():
            action_values = self._q_network_local(net_state)
        self._q_network_local.train()
        return int(argmax(action_values.cpu().data.numpy()))

    def learn(self) -> None:
        # self._steps = (self._steps + 1) % self._update_every_steps
        self._steps = (self._steps + 1) % (self._update_every_steps * self._hard_update_every_steps)
        if self._steps % self._update_every_steps == 0:
            if self._memory.get_current_size() >= self._batch_size:
                states, actions, rewards, next_states, dons, _ = self._memory.sample()

                states = torch.from_numpy(states).float().to(self._device)
                actions = torch.from_numpy(actions).long().to(self._device)
                rewards = torch.from_numpy(rewards).float().to(self._device)
                next_states = torch.from_numpy(next_states).float().to(self._device)
                dons = torch.from_numpy(dons).float().to(self._device)

                # Get max predicted Q values (for next states) from target model
                q_targets_next = self._q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
                # Compute Q targets for current states
                q_targets = rewards + (self._gamma * q_targets_next * (1 - dons))

                # Get expected Q values from local model
                q_expected = self._q_network_local(states).gather(1, actions)

                # Compute loss
                loss = F.mse_loss(q_expected, q_targets)
                # Minimize the loss
                self._optimizer.zero_grad()
                loss.backward()  # type:ignore
                # gradient clipping
                clip_grad_norm_(self._q_network_local.parameters(), 10.0)
                self._optimizer.step()

        if self._steps % self._hard_update_every_steps == 0:
            self._hard_update(self._q_network_local, self._q_network_target, self._tau)

    @staticmethod
    def _hard_update(local_model: Any, target_model: Any, tau: float) -> None:
        """
        Hard update of parameters.
        :param local_model: Any.
        :param target_model: Any.
        :param tau: float.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
class DQNAgentPER(BaseAgent):
    """
    Class for double q network agent with prioritized experience replay.
    """

    def __init__(self, env: Any, actions_dim: int, memory_size: int, batch_size: int, q_network_class: Any, \
                 gamma: float, alpha: float) -> None:
        replay_buffer = PrioritizedReplayBuffer(
            state_dim=env.observation_space.shape[0],
            actions_dim=actions_dim,
            buffer_size=memory_size,
            batch_size=batch_size,
            n_actions=env.action_space.n,
            alpha=alpha
        )
        BaseAgent.__init__(self, env=env, replay_buffer=replay_buffer, gamma=gamma)

        self._q_network_local = q_network_class(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            seed=988
        )
        self._q_network_target = q_network_class(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            seed=988
        )
        self._optimizer = optim.Adam(self._q_network_local.parameters(), lr=5e-4)

        self._steps = 0
        self._batch_size = batch_size

        self._update_every_steps = 2
        self._hard_update_every_steps = 8
        self._tau = 0.001

        self._per_epsilon = 1e-6

    def set_optimizing_parameters(self, update_every_steps: int, hard_update_every_steps: int, tau: float) -> None:
        """
        Sets neural network training parameters.
        :param update_every_steps: int. After how many steps the local neural network should be updated.
        :param hard_update_every_steps: int. After how many steps the targe neural network should be updated.
        :param tau: float. Hard update parameter.
        """
        self._update_every_steps = update_every_steps
        self._hard_update_every_steps = hard_update_every_steps
        self._tau = tau

    def _get_greedy_action(self, state: Any) -> int:
        """
        Gets the greedy action.
        :param state: Any.
        :return: int. Greedy action.
        """
        net_state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._q_network_local.eval()
        with torch.no_grad():
            action_values = self._q_network_local(net_state)
        self._q_network_local.train()
        return int(argmax(action_values.cpu().data.numpy()))

    # pylint: disable=arguments-differ
    def learn(self, beta: float) -> None:  # type:ignore
        """
        :param beta: float. Beta parameter for calculation.
        """
        # self._steps = (self._steps + 1) % self._update_every_steps
        self._steps = (self._steps + 1) % (self._update_every_steps * self._hard_update_every_steps)
        if self._steps % self._update_every_steps == 0:
            if self._memory.get_current_size() >= self._batch_size:
                states, actions, rewards, next_states, dons, _, weights, indices = self._memory.sample(beta)

                states = torch.from_numpy(states).float().to(self._device)
                actions = torch.from_numpy(actions).long().to(self._device)
                rewards = torch.from_numpy(rewards).float().to(self._device)
                next_states = torch.from_numpy(next_states).float().to(self._device)
                dons = torch.from_numpy(dons).float().to(self._device)
                weights = torch.from_numpy(weights).float().to(self._device)

                # Get max predicted Q values (for next states) from target model
                q_targets_next = self._q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
                # Compute Q targets for current states
                # q_targets = rewards + (self._gamma * q_targets_next * (1 - dons))
                q_targets = rewards + (self._gamma * q_targets_next * (1 - dons))

                # Get expected Q values from local model
                q_expected = self._q_network_local(states).gather(1, actions)

                # compute element-wise loss + per importance sampling (PER)
                # loss = F.mse_loss(q_expected, q_targets) # not element
                loss_elements = F.smooth_l1_loss(q_expected, q_targets, reduction="none")
                loss = torch.mean(loss_elements * weights)

                # Minimize the loss
                self._optimizer.zero_grad()
                loss.backward()  # type:ignore
                # gradient clipping
                clip_grad_norm_(self._q_network_local.parameters(), 10.0)
                self._optimizer.step()

                # PER - update priorities
                loss_for_prior = loss_elements.detach().cpu().numpy()
                new_priorities = loss_for_prior + self._per_epsilon
                self._memory.update_priorities(indices, new_priorities)

        if self._steps % self._hard_update_every_steps == 0:
            self._hard_update(self._q_network_local, self._q_network_target, self._tau)

    # pylint: enable=arguments-differ

    @staticmethod
    def _hard_update(local_model: Any, target_model: Any, tau: float) -> None:
        """
        Hard update of parameters.
        :param local_model: Any.
        :param target_model: Any.
        :param tau: float.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# pylint: enable=too-many-instance-attributes
# pylint: enable=no-member
# pylint: enable=too-many-arguments
# pylint: enable=too-many-locals
