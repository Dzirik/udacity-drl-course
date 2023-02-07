"""
Replay buffer.
"""
from typing import Any, Tuple, List, Optional

from numpy import zeros, ndarray, dtype, array
from numpy.random import choice, uniform
from sklearn.preprocessing import OneHotEncoder

from src.external.segment_tree import SumSegmentTree, MinSegmentTree


# pylint: disable=too-many-instance-attributes
class ReplayBuffer:
    """
    Numpy-based replay buffer.
    """

    def __init__(self, state_dim: int, actions_dim: Any, buffer_size: int, batch_size: int, n_actions: int) -> None:
        """
        :param state_dim: int. Dimension of state space.
        :param actions_dim: int. Dimension of actions dim (not the amount of actions!!).
        :param buffer_size: int. Size of the buffer's memory.
        :param batch_size: int. Size of the batch generated.
        :param n_actions: int. Number of distinct actions. Used for one hot encoding equivalence for actions if
                               actions_dim equals 1.
        """
        self._states_buffer = zeros((buffer_size, state_dim))
        self._actions_buffer = zeros((buffer_size, actions_dim))
        self._rewards_buffer = zeros((buffer_size, 1))
        self._next_states_buffer = zeros((buffer_size, state_dim))
        self._done_buffer = zeros((buffer_size, 1))

        self._buffer_size = buffer_size
        self._batch_size = batch_size

        self._pointer: int = 0
        self._current_size: int = 0

        self._ooh = OneHotEncoder(sparse=False)
        self._do_ooh = False
        if actions_dim == 1 and n_actions > 0:
            self._do_ooh = True
            training_data: ndarray[Any, dtype[Any]] = array(list(range(n_actions))).reshape((n_actions, 1))
            self._ooh.fit(training_data)

    def add(self, state: ndarray[Any, dtype[Any]], action: ndarray[Any, dtype[Any]], reward: float, \
            next_state: ndarray[Any, dtype[Any]], done: bool) -> None:
        """
        Adds the experience set.
        :param state: ndarray[Any, dtype[Any]].
        :param action: ndarray[Any, dtype[Any]].
        :param reward: float.
        :param next_state: ndarray[Any, dtype[Any]].
        :param done: bool.
        :return:
        """
        self._states_buffer[self._pointer, :] = state
        self._actions_buffer[self._pointer, :] = action
        self._rewards_buffer[self._pointer, :] = reward
        self._next_states_buffer[self._pointer, :] = next_state
        self._done_buffer[self._pointer, :] = done

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._current_size = min(self._current_size + 1, self._buffer_size)

    def sample(self) -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                              ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], Optional[ndarray[Any, dtype[Any]]]]:
        """
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                              ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], Optional[ndarray[Any, dtype[Any]]]]:
                 (states, actions, rewards, next_states, dons, actions_oh).
        """
        indices = choice(self._current_size, size=self._batch_size, replace=False)
        actions = self._actions_buffer[indices, :]
        actions_ooh = None
        if self._do_ooh:
            actions_ooh = self._ooh.transform(actions)
        return (
            self._states_buffer[indices, :],
            actions,
            self._rewards_buffer[indices, :],
            self._next_states_buffer[indices, :],
            self._done_buffer[indices, :],
            actions_ooh
        )

    def get_current_size(self) -> int:
        """
        Gets the current size.
        """
        return self._current_size


# pylint: enable=too-many-instance-attributes

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Numpy-based prioritized experience buffer.

    Segment trees are used here. Please see the file with code for more information.

    Concept was taken from here: https://github.com/Curt-Park/rainbow-is-all-you-need.
    """

    def __init__(self, state_dim: int, actions_dim: Any, buffer_size: int, batch_size: int, n_actions: int, \
                 alpha: float) -> None:
        """
        :param state_dim: int. Dimension of state space.
        :param actions_dim: int. Dimension of actions dim (not the amount of actions!!).
        :param buffer_size: int. Size of the buffer's memory.
        :param batch_size: int. Size of the batch generated.
        :param n_actions: int. Number of distinct actions. Used for one hot encoding equivalence for actions if
                               actions_dim equals 1.
        :param alpha: float. Prioritisation alpha parameter.
        """
        ReplayBuffer.__init__(self, state_dim, actions_dim, buffer_size, batch_size, n_actions)

        self._alpha = alpha
        self._tree_pointer = 0
        self._max_priority = 1.0

        # computing tree capacity (has to be power of 2)
        self._tree_capacity = 1
        while self._tree_capacity < self._buffer_size:
            self._tree_capacity = self._tree_capacity * 2

        self._sum_tree = SumSegmentTree(capacity=self._tree_capacity)
        self._min_tree = MinSegmentTree(capacity=self._tree_capacity)

    def add(self, state: ndarray[Any, dtype[Any]], action: ndarray[Any, dtype[Any]], reward: float, \
            next_state: ndarray[Any, dtype[Any]], done: bool) -> None:
        """
        Adds the experience set.
        :param state: ndarray[Any, dtype[Any]].
        :param action: ndarray[Any, dtype[Any]].
        :param reward: float.
        :param next_state: ndarray[Any, dtype[Any]].
        :param done: bool.
        """
        super().add(state, action, reward, next_state, done)

        self._sum_tree[self._tree_pointer] = self._max_priority ** self._alpha
        self._min_tree[self._tree_pointer] = self._max_priority ** self._alpha
        self._tree_pointer = (self._tree_pointer + 1) % self._buffer_size

    def _sample_indices(self) -> List[int]:
        """
        Sample the indices proportionally to the distribution of the priorities.
        :return: List[int]. List of indices.
        """
        indices = []
        distribution_mass = self._sum_tree.sum(0, self._tree_capacity - 1) # this caused an error
        # distribution_mass = self._sum_tree.sum(0, self._buffer_size - 1)
        segment_mass = distribution_mass / self._batch_size

        for i in range(self._batch_size):
            lower_segment_bound = segment_mass * i
            upper_segment_bound = segment_mass * (i + 1)
            upper_bound = uniform(lower_segment_bound, upper_segment_bound)
            index = self._sum_tree.find_prefixsum_idx(upper_bound)
            # indices.append(index) # just adding the index caused an error in a corner case
            if index < self._buffer_size:
                indices.append(index)
            else:
                indices.append(self._current_size)

        return indices

    def _calculate_weight(self, index: int, beta: float) -> float:
        """
        Calculates the weights.
        :param index: int. Index of the experience for which it has to be calculated.
        :param beta: float. Beta parameter for calculation.
        :return: float. Weight.
        """
        min_probability = self._min_tree.min() / self._sum_tree.sum()
        max_weight = (min_probability * self._tree_capacity) ** (-beta)

        experience_probability = self._sum_tree[index] / self._sum_tree.sum()
        weight = (experience_probability * self._tree_capacity) ** (-beta)

        return float(weight / max_weight)

    # pylint: disable=arguments-differ
    def sample(self, beta: float) -> Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]],  # type:ignore
                                           ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]],
                                           Optional[ndarray[Any, dtype[Any]]], Optional[ndarray[Any, dtype[Any]]], List[
                                               int]]:
        """
        Sample the batch from the buffer.
        :param beta: float. Beta parameter for calculation.
        :return: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], \
                              ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], Optional[ndarray[Any, dtype[Any]]]
                              List[int]]:
                 (states, actions, rewards, next_states, dons, actions_oh, weights, indices).
        """
        indices = self._sample_indices()
        actions = self._actions_buffer[indices, :]
        actions_ooh = None
        if self._do_ooh:
            actions_ooh = self._ooh.transform(actions)
        weights: ndarray[Any, dtype[Any]] = array([self._calculate_weight(index, beta) for index in indices]).reshape(
            (self._batch_size, 1))
        return (
            self._states_buffer[indices, :],
            actions,
            self._rewards_buffer[indices, :],
            self._next_states_buffer[indices, :],
            self._done_buffer[indices, :],
            actions_ooh,
            weights,
            indices
        )

    # pylint: enable=arguments-differ

    def update_priorities(self, indices: List[int], priorities: ndarray[Any, dtype[Any]]) -> None:
        """
        Updates the priorities in the tree.
        :param indices: List[int]. List of indices to be updated.
        :param priorities: ndarray[Any, dtype[Any]]. (k, 1) array.
        """
        for index, priority in zip(indices, priorities):
            self._sum_tree[index] = priority ** self._alpha
            self._min_tree[index] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
