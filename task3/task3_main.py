""" Kuhn Poker

    Actions:
        - 0: 'pass'
        - 1: 'Bet'
    Information State:
        - '0': Player 0 turn
        - '1p': Player 1 turn, Player 0 passed
        -

    state.information_state_tensor: This would be an input to a neural network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import matplotlib.pyplot as plt

import pyspiel
from open_spiel.python import rl_environment, policy
from open_spiel.python.algorithms import exploitability_descent, eva, exploitability

import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from absl import app, logging, flags

import settings

# Temporarily disable TF2 until we update the code.
tf.disable_v2_behavior()

# own imports
import models


def example_game():
    """
        Print an example of the Actions and Information state tensor of the Kuhn Poker game
    """
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state()
    print("Initial deck probabilities: ")
    print(state.chance_outcomes())

    # DEALER (random generator)
    rng = np.random.default_rng()
    cards = rng.choice(3, size=2, replace=False)  # generate two unique numbers/cards

    print("DEAL player 0: " + state.action_to_string(cards[0]))
    state.apply_action(cards[0])  # Deal first card
    print("DEAL player : " + state.action_to_string(cards[1]))
    state.apply_action(cards[1])  # Deal second card

    round = 0
    while state.is_terminal() == False:
        print("Start Round " + str(round) + ": " + state.information_state_string())
        print("information tensor: ")
        tensor = state.information_state_tensor()
        player_id = tensor[:2]
        card_id = tensor[2:5]
        player0_action = tensor[5:7]
        player1_action_response = tensor[7:9]
        player0_action_response = tensor[9:]
        print("     player id   : " + str(player_id))
        print("     card id         : " + str(card_id))
        print("     player 0[P,B]   : " + str(player0_action))
        print("     player 1[P,B]   : " + str(player1_action_response))
        print("     player 0[P,B]   : " + str(player0_action_response))

        action_p = np.random.randint(2)
        print("Applying action player " + str(state.current_player()) + ": " + state.action_to_string(action_p))
        state.apply_action(action_p)

        round += 1

    print(state)



if __name__ == "__main__":
    # An example of the Kuhn Poker game
    #example_game()

    fig = plt.figure()
    ax = fig.add_subplot()
    # CFR tab
    results_cfr_tab = models.cfr_tabular()
    ax.plot(results_cfr_tab)

    # CFR NN
    results_cfr_nn = models.cfr_nn()
    ax.plot(results_cfr_nn)

    # tabular ED
    results_ed_tab = models.tabular_ED()
    ax.plot(results_ed_tab)

    # neural network ED
    results_ed_nn = models.nn_ED()
    ax.plot(results_ed_nn)

    ax.set_title("NashConv during learning")
    ax.set_xlabel("Iterations")
    ax.set_xscale('log')
    ax.set_ylabel("NashConv")
    ax.set_yscale('log')
    plt.legend(["CFR-tab", "CFR-NN", "ED-tab", "ED-NN"])
    plt.show()

