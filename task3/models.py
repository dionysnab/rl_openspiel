""" MODELS.py

    This file contains the functions called to train a policy depending on the optimization technique
    and the structure of the model (neural network or tabular)

"""
import time

import numpy as np
import pyspiel
from open_spiel.python import rl_environment, policy
from open_spiel.python.algorithms import exploitability_descent, eva, exploitability, cfr, deep_cfr

import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from absl import app, logging, flags

import settings

# Temporarily disable TF2 until we update the code.
tf.disable_v2_behavior()

def tabular_ED():
    """

    """
    print("Learning tabular ED...")

    args = settings.dictEDTab()
    game = pyspiel.load_game(args['game_name'])
    solver = exploitability_descent.Solver(game)

    nash_conv_list = np.zeros((args['num_steps'],))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args['num_steps']):
            conv = solver.step(sess, args['init_lr']*np.power(args['exp_rate'], step/args['num_steps']))
            nash_conv_list[step] = conv
            if step % args['print_freq'] == 0:
                print("Iteration {} conv {}".format(step, conv))

    return nash_conv_list


def nn_ED():
    """

    """
    print("Learning Neural net ED...")
    args = settings.dictEDNN()
    # Create the game to use, and a loss calculator for it
    game = pyspiel.load_game(args['game_name'])

    # ED loss for network training
    loss_calculator = exploitability_descent.LossCalculator(game)

    # Build the network
    num_hidden = args['num_hidden']
    num_layers = args['num_layers']
    layer = tf.constant(loss_calculator.tabular_policy.state_in, tf.float64)
    for _ in range(num_layers):
        regularizer = (tf.keras.regularizers.l2(l=args['regularizer_scale']))
        layer = tf.layers.dense(
            layer, num_hidden, activation=tf.nn.relu,
            kernel_regularizer=regularizer)
    regularizer = (tf.keras.regularizers.l2(l=args['regularizer_scale']))
    layer = tf.layers.dense(
        layer, game.num_distinct_actions(), kernel_regularizer=regularizer)

    # ED loss: softmax layer -> output of network is tabular policy
    tabular_policy = loss_calculator.masked_softmax(layer)

    nash_conv, loss = loss_calculator.loss(tabular_policy)
    loss += tf.losses.get_regularization_loss()

    # Use a simple gradient descent optimizer
    learning_rate = tf.placeholder(tf.float64, (), name="learning_rate")
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer_step = optimizer.minimize(loss)

    # Store nashConv metric
    nash_conv_list = np.zeros((args['num_steps'],))
    # Training loop
    with tf.train.MonitoredTrainingSession() as sess:
        for step in range(args['num_steps']):
            t0 = time.time()
            conv, _ = sess.run(
                [nash_conv, optimizer_step],
                feed_dict={
                    learning_rate: args['init_lr'] * np.power(args['exp_rate'], step/args['num_steps']),
                })
            t1 = time.time()

            nash_conv_list[step] = conv
            # Optionally log our progress
            if step % args['print_freq'] == 0:
                print("Iteration {} conv {}".format(step, conv))

    return nash_conv_list


def cfr_tabular():
    """

    """

    args = settings.dictCFRTab()
    game = pyspiel.load_game(args['game'],
                             {"players": pyspiel.GameParameter(args['players'])})
    cfr_solver = cfr.CFRSolver(game)
    nashconv_list = np.zeros((args['iterations'],))
    for i in range(args['iterations']):
        cfr_solver.evaluate_and_update_policy()
        conv = exploitability.nash_conv(game, cfr_solver.average_policy())
        nashconv_list[i] = conv
        if i % args['print_freq'] == 0:
            print("Iteration {} conv {}".format(i, conv))

    return nashconv_list

def cfr_nn():
    """

    """
    args = settings.dictCFRNN()
    ""
    game = pyspiel.load_game(args['game'])
    with tf.Session() as sess:
        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers=(16,),
            advantage_network_layers=(16,),
            num_iterations=1,
            num_traversals=1,
            learning_rate=args['init_lr'],
            batch_size_advantage=128,
            batch_size_strategy=1024,
            memory_capacity=1e7,
            policy_network_train_steps=1,
            advantage_network_train_steps=1,
            reinitialize_advantage_networks=False)

        sess.run(tf.global_variables_initializer())
        nashconv_list = np.zeros((args['num_steps'],))
        for step in range(args['num_steps']):
            #Solve step (1 iteration)
            _, advantage_losses, policy_loss = deep_cfr_solver.solve()
            average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
            conv = exploitability.nash_conv(game, average_policy)
            nashconv_list[step] = conv
            if step % args['log_freq'] == 0:
                print("Iteration {} NashConv {}".format(step, conv))


    return nashconv_list
