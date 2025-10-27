import numpy as np
import gymnasium as gym
from collections import defaultdict

class BlackJackAgent:
  def __init__(
    self,
    env: gym.Env,
    learning_rate: float,
    initial_epsilon: float,
    epsilon_decay: float,
    final_epsilon: float,
    discount_factor: float = 0.95
  ):
    """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
    """
    self.q_vals = defaultdict(lambda: np.zeros(env.action_space.n))
    self.env = env

    self.discount_factor = discount_factor
    self.learning_rate = learning_rate

    self.epsilon = initial_epsilon
    self.final_epsilon = final_epsilon
    self.epsilon_decay = epsilon_decay

    self.training_error = []

  def get_action(self, obs: tuple[int, int, bool]) -> int:
    """Choose an action using epsilon-greedy strategy.

    Returns:
      action: 0 (stand) or 1 (hit)
    """
    if np.random.random() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return int(np.argmax(self.q_vals[obs]))
    
  def update(
    self,
    obs: tuple[int, int, bool],
    action: int, 
    reward: float,
    terminated: bool,
    next_obs: tuple[int, int, bool]
  ):
    """
    Update Q-value based on experience.
    """
    
    future_q_val = (not terminated) * np.max(self.q_vals[next_obs])
    
    # Bellman equation
    target = reward + self.discount_factor * future_q_val
    temporal_difference = target - self.q_vals[obs][action]

    self.q_vals[obs][action] += self.learning_rate * temporal_difference

    self.training_error.append(temporal_difference)

  def decay_epsilon(self):
    self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)