""" Rock Paper Sciccors"""
"TODO: Find out how Q-learning works in a Simultaneous-Move game"
import random
import pyspiel
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from open_spiel.python.egt import alpharank, heuristic_payoff_table, utils
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.algorithms import tabular_qlearner, random_agent, dqn
from open_spiel.python import rl_environment
from open_spiel.python.egt.utils import game_payoffs_array


def print_loss(losses, title):
    episodes = np.indices((len(losses[0]),1))

    fig = plt.figure()

    ax_agent1 = fig.add_subplot(121)
    ax_agent2 = fig.add_subplot(122)

    ax_agent1.plot( losses[0])
    ax_agent2.plot( losses[1])
    plt.show()


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

def train_agent(training_episodes, agents, random_agents, env):
    probs = []
    loss_agent1 = []
    loss_agent2 = []
    for cur_episode in range(training_episodes):
        #if cur_episode % int(1e4) == 0:
        #    win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
        #    print("Starting episode " + str(cur_episode) + " win_rates " + str(win_rates))
        time_step = env.reset()  # Start a new sequence and return first timestep of this sequence
        agent_output1 = agents[0].step(time_step, is_evaluation=False)
        agent_output2 = agents[1].step(time_step, is_evaluation=False)
        time_step = env.step([agent_output1.action, agent_output2.action])


        for agent in agents:
            agent.step(time_step)

        loss_agent1.append(agents[0]._last_loss_value)
        loss_agent2.append(agents[1]._last_loss_value)
    return agents, [loss_agent1, loss_agent2]

def print_payoff_tables(game_names):
    for game_name in game_names:
        game = pyspiel.load_game(game_name)
        payoff_matrici = game_payoffs_array(game)
        print("Game: " + game_name)
        print("payoff_matrix Player 1: ")
        print(payoff_matrici[0])
        print("payoff_matrix Player 2: ")
        print(payoff_matrici[1])


# GAME CONFIGURATION
training_episodes = 1000
games = ["matrix_rps", "matrix_mp", "matrix_sh", "matrix_cd"]
num_players = 2

# DQN parameters
hidden_layers_sizes = [32, 32]
replay_buffer_capacity = int(1e4)


def play_game(game_name):
    # define RL environment & game details
    env = rl_environment.Environment(game_name)
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

    # 1. Selfplay tab_ql agents
    tab_ql_agents, losses= train_agent(training_episodes, tab_ql_agents, random_agents, env)
    # 3. Crossplay tab_ql vs random
    win_rates_final = eval_against_random_bots(env, tab_ql_agents, random_agents, 1000)
    print("Final win_rates (agent1, agent2):   ", win_rates_final)
    print_loss(losses, "title")


if __name__ == "__main__":

    print_payoff_tables(games)

    play_game(games[3])


