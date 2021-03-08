""" Rock Paper Sciccors"""
"TODO: Find out how Q-learning works in a Simultaneous-Move game"
import random
import pyspiel
import numpy as np
import tensorflow as tf
import logging

from open_spiel.python.egt import alpharank, heuristic_payoff_table, utils
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.algorithms import tabular_qlearner, random_agent, dqn
from open_spiel.python import rl_environment


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  wins = np.zeros(2)
  for player_pos in range(2):
      if player_pos == 0:
          cur_agents = [trained_agents[0], random_agents[1]]
      else:
          cur_agents = [trained_agents[1], random_agents[0]]
      for _ in range(num_episodes):
          time_step = env.reset()
          agent_output1 = cur_agents[0].step(time_step, is_evaluation=True)
          agent_output2 = cur_agents[1].step(time_step, is_evaluation=True)
          time_step = env.step([agent_output1.action, agent_output2.action])

          """while not time_step.last():
              player_id = time_step.observations["current_player"]
              agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
              time_step = env.step([agent_output.action])
          """
          if time_step.rewards[player_pos] > 0:
              wins[player_pos] += 1

  return wins/ num_episodes

def eval_against_agent(env, trained_agents, num_episodes):
    wins = np.zeros(2)

    for _ in range(num_episodes):
        time_step = env.reset()
        agent_output1 = trained_agents[0].step(time_step, is_evaluation=True)
        agent_output2 = trained_agents[1].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output1.action, agent_output2.action])

        for player_pos in range(2):
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1

    return wins / num_episodes

def train_agent(training_episodes, agents, random_agents):
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
            print("Starting episode " + str(cur_episode) + " win_rates " + str(win_rates))
        time_step = env.reset()  # Start a new sequence and return first timestep of this sequence
        agent_output1 = agents[0].step(time_step, is_evaluation=True)
        agent_output2 = agents[1].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output1.action, agent_output2.action])
        """
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])
        """
        for agent in agents:
            agent.step(time_step)
    return agents

# GAME CONFIGURATION
training_episodes = 100000
games = ["matrix_rps", "matrix_mp", "matrix_sh", "matrix_cd"]
num_players = 2

# DQN parameters
hidden_layers_sizes = [32, 32]
replay_buffer_capacity = int(1e4)

if __name__ == "__main__":

    for game in games:
        # define RL environment & game details
        env = rl_environment.Environment(game)
        num_actions = env.action_spec()["num_actions"]
        state_size = env.observation_spec()["info_state"][0]

        print("playing matrix game: " + env.name)

        # tabular_qlearner agents
        tab_ql_agents = [
            tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        # Random agents (benchmark)
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        # start tf session for dqn agents
        with tf.compat.v1.Session() as sess:
            dqn_agents = [
                dqn.DQN(
                    sess,
                    player_id=0,
                    state_representation_size=state_size,
                    num_actions=num_actions,
                    hidden_layers_sizes=hidden_layers_sizes,
                    replay_buffer_capacity=replay_buffer_capacity
                )
                for idx in range(num_players)
            ]
            sess.run(tf.compat.v1.global_variables_initializer())

            # 1. Selfplay tab_ql agents
            tab_ql_agents = train_agent(training_episodes, tab_ql_agents, random_agents)

            # 2. Selfplay dqn agents
            dqn_agents = train_agent(training_episodes, dqn_agents, random_agents)

            # 3. Crossplay tab_ql vs dqn
            win_rates_final = eval_against_agent(env, [tab_ql_agents[0], dqn_agents[0]], 1000)
            print("Final win_rates (tab_ql, dqn):   ", win_rates_final)

