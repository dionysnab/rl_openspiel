""" Dynamics of learning in benchmark matrix games"""


"""
    Reference: Bloembergen, D., Tuyls, K., Hennes, D., & Kaisers, 
            M. (2015). Evolutionary dynamics of multi-agent learning: A survey. 
            Journal of Artificial Intelligence Research, 53, 659-697
            
    Example class of Replicator dynamics in python/egt/dynamics.py (Openspiel)
"""
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from open_spiel.python.egt import dynamics, visualization, visualization_test
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel
import numpy as np
import dynamics_self
# Config
games = ["matrix_rps", "matrix_mp", "matrix_sh", "matrix_cd"]
temp = 0


def plot_quiver(size=2, dynamics=None, title="", actions=None):
    fig = plt.figure()

    if size == 3:
        ax = fig.add_subplot(111, projection="3x3")
    else:
        ax = fig.add_subplot(111, projection="2x2")

    ax.set_title(title)
    res = ax.quiver(dynamics)
    if actions!=None:
        if size == 2:
            ax.set_xlabel(actions[0])
            ax.set_ylabel(actions[1])
        # TODO: how to label 3x3 axes
    plt.show()

def plot_streamplot(size=2, dynamics=None, title="", actions = None):
    '''
    Args:
        size : 2x2 or 3x3 plot
        dynamics : list of dynamics (list of np.arrays)
        title : list of titles for plot
        actions : list of string xlabel, ylabel, zlabel

    '''
    fig = plt.figure()

    # TODO: Make function with simple loop
    if len(dynamics) == 1: # one simply plot
        if size == 3:
            ax = fig.add_subplot(111, projection="3x3")
        else:
            ax = fig.add_subplot(111, projection="2x2")

        ax.set_title(title[0])
        res = ax.streamplot(dynamics[0])

        if actions != None:
            if size == 2:
                ax.set_xlabel(actions[0])
                ax.set_ylabel(actions[1])
            # TODO: how to label 3x3 axes

    elif len(dynamics) == 2: # comparison of Replicator vs Lenient-boltzmannq
        if size == 3:
            ax_rep = fig.add_subplot(121, projection="3x3")
            ax_boltz = fig.add_subplot(122, projection="3x3")
        else :
            ax_rep = fig.add_subplot(121, projection="2x2")
            ax_boltz = fig.add_subplot(122, projection="2x2")

        ax_rep.set_title(title[0])
        ax_boltz.set_title(title[1])
        res_rep = ax_rep.streamplot(dynamics[0])
        res_boltz = ax_boltz.streamplot(dynamics[1])

    elif len(dynamics) == 3: # Comparison of temperature in lenient-boltzmannq
        if size == 3:
            ax_temp1 = fig.add_subplot(131, projection="3x3")
            ax_temp2 = fig.add_subplot(132, projection="3x3")
            ax_temp3 = fig.add_subplot(133, projection="3x3")
        else :
            ax_temp1 = fig.add_subplot(131, projection="2x2")
            ax_temp2 = fig.add_subplot(132, projection="2x2")
            ax_temp3 = fig.add_subplot(133, projection="2x2")

        ax_temp1.set_title(title[0])
        ax_temp2.set_title(title[1])
        ax_temp3.set_title(title[2])
        res_temp1 = ax_temp1.streamplot(dynamics[0])
        res_temp2 = ax_temp2.streamplot(dynamics[1])
        res_temp3 = ax_temp3.streamplot(dynamics[2])

    plt.show()


def explore_lenient_temp(game_name):
    game = pyspiel.load_game(game_name)
    payoff_matrix = game_payoffs_array(game)
    temps = [0.1, 0.5, 1]

    dyn_list = []
    title_list = []
    for i in range(len(temps)):
        temp = temps[i]
        title = "temp = " + str(temp)

        if game_name == "matrix_rps":
            size = 3
            dyn = dynamics_self.SinglePopulationDynamics(payoff_matrix, dynamics_self.boltzmannq, temp)
        else:
            size = 2
            dyn = dynamics_self.MultiPopulationDynamics(payoff_matrix, dynamics_self.boltzmannq, temp)
        dyn_list.append(dyn)
        title_list.append(title)

    plot_streamplot(size, dyn_list, title_list)


def compare_replicator_boltzmann(game_name):
    game = pyspiel.load_game(game_name)
    payoff_matrix = game_payoffs_array(game)
    print('payoff matrix player 1: ')
    print(payoff_matrix[0])
    print('payoff matrix player 2: ')
    print(payoff_matrix[1])

    if game_name == "matrix_rps":
        dyn_rep = dynamics_self.SinglePopulationDynamics(payoff_matrix, dynamics_self.replicator)
        dyn_boltz = dynamics_self.SinglePopulationDynamics(payoff_matrix, dynamics_self.boltzmannq)
    else:
        dyn_rep = dynamics_self.MultiPopulationDynamics(payoff_matrix, dynamics_self.replicator)
        dyn_boltz = dynamics_self.MultiPopulationDynamics(payoff_matrix, dynamics_self.boltzmannq)
    title_rep = game_name + ": Replicator Dynamics"
    title_boltz = game_name + ": Lenient-Boltzmannq"

    dynamics_list = [dyn_rep, dyn_boltz]
    titles = [title_boltz, title_rep]




if __name__ == "__main__":
    # TEST X: Leniet Boltzmann Q-learning - influence of temperature
    test_game_lenient_boltz_temp = "matrix_mp"
    explore_lenient_temp(test_game_lenient_boltz_temp)
    '''
    for game_name in games:
        game = pyspiel.load_game(game_name)
        payoff_matrix = game_payoffs_array(game)
        print('payoff matrix player 1: ')
        print(payoff_matrix[0])
        print('payoff matrix player 2: ')
        print(payoff_matrix[1])

        if game_name == "matrix_rps":
            dyn_rep = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
            #dyn_boltz = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.boltzmannq)
            dyn_boltz = dynamics.SinglePopulationDynamics(payoff_matrix, lenient_boltzmannq)
        else:
            dyn_rep = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)
            #dyn_boltz = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.boltzmannq)
            dyn_boltz = dynamics.MultiPopulationDynamics(payoff_matrix, lenient_boltzmannq)

        fig = plt.figure()
        # fig, axs = plt.subplots(2, 1)

        # fig = Figure(figsize=(4, 4))
        if payoff_matrix.shape[1] == 3:
            ax_rep = fig.add_subplot(121, projection="3x3")
            ax_boltz = fig.add_subplot(122, projection="3x3")
        else:
            ax_rep = fig.add_subplot(121, projection="2x2")
            ax_boltz = fig.add_subplot(122, projection="2x2")

        ax_rep.set_title(game_name + " replicator")
        ax_boltz.set_title(game_name + " boltzmann")
        res_rep = ax_rep.quiver(dyn_rep)
        res_boltz = ax_boltz.quiver(dyn_boltz)
        plt.show()
        '''
