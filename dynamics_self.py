from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def replicator(state, fitness):
  """Continuous-time replicator dynamics.

  This is the standard form of the continuous-time replicator dynamics also
  known as selection dynamics.

  For more details, see equation (5) page 9 in
  https://jair.org/index.php/jair/article/view/10952

  Args:
    state: Probability distribution as an `np.array(shape=num_strategies)`.
    fitness: Fitness vector as an `np.array(shape=num_strategies)`.

  Returns:
    Time derivative of the population state.
  """
  avg_fitness = state.dot(fitness)
  return state * (fitness - avg_fitness)


def boltzmannq(state, fitness, temperature=1.):
  """Selection-mutation dynamics modeling Q-learning with Boltzmann exploration.

  For more details, see equation (10) page 15 in
  https://jair.org/index.php/jair/article/view/10952

  Args:
    state: Probability distribution as an `np.array(shape=num_strategies)`.
    fitness: Fitness vector as an `np.array(shape=num_strategies)`.
    temperature: A scalar parameter determining the rate of exploration.

  Returns:
    Time derivative of the population state.
  """
  exploitation = (1. / temperature) * replicator(state, fitness)
  exploration = (np.log(state) - state.dot(np.log(state).transpose()))
  return exploitation - state * exploration


class SinglePopulationDynamics(object):
  """Continuous-time single population dynamics.

  Attributes:
    payoff_matrix: The payoff matrix as an `numpy.ndarray` of shape `[2, k_1,
      k_2]`, where `k_1` is the number of strategies of the first player and
      `k_2` for the second player. The game is assumed to be symmetric.
    dynamics: A callback function that returns the time-derivative of the
      population state.
  """

  def __init__(self, payoff_matrix, dynamics, temp):
    """Initializes the single-population dynamics."""
    assert payoff_matrix.ndim == 3
    assert payoff_matrix.shape[0] == 2
    assert np.allclose(payoff_matrix[0], payoff_matrix[1].T)
    self.payoff_matrix = payoff_matrix[0]
    self.dynamics = dynamics
    self.temp = temp

  def __call__(self, state=None, time=None):
    """Time derivative of the population state.

    Args:
      state: Probability distribution as list or
        `numpy.ndarray(shape=num_strategies)`.
      time: Time is ignored (time-invariant dynamics). Including the argument in
        the function signature supports numerical integration via e.g.
        `scipy.integrate.odeint` which requires that the callback function has
        at least two arguments (state and time).

    Returns:
      Time derivative of the population state as
      `numpy.ndarray(shape=num_strategies)`.
    """
    state = np.array(state)
    assert state.ndim == 1
    assert state.shape[0] == self.payoff_matrix.shape[0]
    # (Ax')' = xA'
    fitness = np.matmul(state, self.payoff_matrix.T)
    return self.dynamics(state, fitness, self.temp)


class MultiPopulationDynamics(object):
  """Continuous-time multi-population dynamics.

  Attributes:
    payoff_tensor: The payoff tensor as an numpy.ndarray of size `[n, k0, k1,
      k2, ...]`, where n is the number of players and `k0` is the number of
      strategies of the first player, `k1` of the second player and so forth.
    dynamics: List of callback functions for the time-derivative of the
      population states, where `dynamics[i]` computes the time-derivative of the
      i-th player's population state. If at construction, only a single callback
      function is provided, the same function is used for all populations.
  """

  def __init__(self, payoff_tensor, dynamics, temp):
    """Initializes the multi-population dynamics."""
    if isinstance(dynamics, list) or isinstance(dynamics, tuple):
      assert payoff_tensor.shape[0] == len(dynamics)
    else:
      dynamics = [dynamics] * payoff_tensor.shape[0]
    self.payoff_tensor = payoff_tensor
    self.dynamics = dynamics
    self.temp = temp

  def __call__(self, state, time=None):
    """Time derivative of the population states.

    Args:
      state: Combined population state for all populations as a list or flat
        `numpy.ndarray` (ndim=1). Probability distributions are concatenated in
        order of the players.
      time: Time is ignored (time-invariant dynamics). Including the argument in
        the function signature supports numerical integration via e.g.
        `scipy.integrate.odeint` which requires that the callback function has
        at least two arguments (state and time).

    Returns:
      Time derivative of the combined population state as `numpy.ndarray`.
    """
    state = np.array(state)
    n = self.payoff_tensor.shape[0]  # number of players
    ks = self.payoff_tensor.shape[1:]  # number of strategies for each player
    assert state.shape[0] == sum(ks)

    states = np.split(state, np.cumsum(ks)[:-1])
    dstates = [None] * n
    for i in range(n):
      # move i-th population to front
      fitness = np.moveaxis(self.payoff_tensor[i], i, 0)
      # marginalize out all other populations
      for i_ in set(range(n)) - {i}:
        fitness = np.tensordot(states[i_], fitness, axes=[0, 1])
      dstates[i] = self.dynamics[i](states[i], fitness, self.temp)

    return np.concatenate(dstates)