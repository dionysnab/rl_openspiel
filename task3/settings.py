""" SETTINGS
    settings.py contains the required training parameters for task 3.
    These parameters are defines as flags.
"""

num_steps = int(1e5)
print_freq = int(num_steps/10)
init_lr = 0.1
exp_rate = 0.96
def dictEDNN():
    """

    """
    dict = {
        "num_steps": num_steps,
        "game_name": "kuhn_poker",
        "print_freq": print_freq,
        "init_lr": init_lr,
        "exp_rate": exp_rate,
        "regularizer_scale": 0.001,
        "num_hidden": 64,
        "num_layers": 3
    }
    return dict


def dictEDTab():
    """

    """
    dict = {
        "num_steps": num_steps,
        "game_name": "kuhn_poker",
        "print_freq": print_freq,
        "init_lr": init_lr,
        "exp_rate": exp_rate
    }
    return dict


def dictCFRTab():
    dict = {
        "iterations": num_steps,
        "game": "kuhn_poker",
        "players": 2,
        "print_freq": print_freq
    }
    return dict


def dictCFRNN():
    """

    """
    dict = {
        "num_steps": num_steps,
        "game": "kuhn_poker",
        "log_freq": print_freq,
        "init_lr": init_lr,
        "regularizer_scale": 0.001,
    }
    return dict