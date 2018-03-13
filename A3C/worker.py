import tensorflow as tf
import threading
import numpy as np
from A3C.network import PolicyValueNetwork
import gym
from matplotlib import pyplot as plt
import collections
import os
import sys
import itertools


def copy_network(from_scope, to_scope):

    global_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    local_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    copy_val = []
    for g, w in zip(local_val, global_val):
        op = g.assign(w)
        copy_val.append(op)

    return copy_val


class Worker():
    def __init__(self, game, id, t_max, num_actions, global_network):

        self.value_state = []
        self.state_buffer = []
        self.state = []
        self.reward = []
        self.episode_state = []
        # current steps of the worker
        self.steps_worker = 0
        self.game = game
        self.id = id
        self.t_max = t_max
        self.global_network = global_network

        # Initialise the environment
        self.env = gym.envs.make(self.game)
        # Size of the action space
        self.num_actions = num_actions

        # Worker network
        with tf.variable_scope(self.id):
            self.w_network = PolicyValueNetwork(self.num_actions, self.id)

        # Cannot change uninitialised graph concurrently.
        self.copy_network = copy_network("global", self.id)

    def play(self, coord, sess):
        with sess.as_default(), sess.graph.as_default():

            observation = self.env.reset()

            # if threading.current_thread().name == "Worker_1":
            #     values_global = sess.run(self.global_network.variables_names)
            #     for k, v in zip(self.global_network.variables_names, values_global):
            #         print("Variable: ", k)
            #         print("Shape: ", v.shape)
            #         print(v)
            #
            #     values_local = sess.run(self.w_network.variables_names)
            #     for k, v in zip(self.w_network.variables_names, values_local):
            #         print("Variable: ", k)
            #         print("Shape: ", v.shape)
            #         print(v)
            #         # print("here 2")
            #     print("here now")
            #
            #     sess.run(copy_network('global', self.id))
            #
            #     values_global = sess.run(self.global_network.variables_names)
            #     for k, v in zip(self.global_network.variables_names, values_global):
            #         print("Variable: ", k)
            #         print("Shape: ", v.shape)
            #         print(v)
            #
            #     values_local = sess.run(self.w_network.variables_names)
            #     for k, v in zip(self.w_network.variables_names, values_local):
            #         print("Variable: ", k)
            #         print("Shape: ", v.shape)
            #         print(v)
            #
            #         # print(values_local)
            #         # print(values_global)
            #         # sess.run(copy_network(self.global_network.variables_names, self.w_network.variables_names))

            while not coord.should_stop():

                t = 0

                sess.run(self.copy_network)

                if self.steps_worker < 4:
                    # Select an random action
                    action = self.env.action_space.sample()
                    # Interact with the environment
                    observation, reward, done, info = self.env.step(action)
                    # process the observation
                    proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                    # reshape from [1,84,84,1] to [84,84]
                    proccessed_state = np.reshape(proccessed_state, [84, 84])
                    # append the processed state to the end of the list
                    self.state.append(proccessed_state)
                    self.reward.append(reward)
                    self.steps_worker +=1
                else:
                    for t in range(self.t_max):
                        action_prob = sess.run([self.w_network.policy],
                                               {self.w_network.state: np.reshape(self.state, [1, 84, 84, 4])})
                        action = np.argmax(action_prob, axis=1)
                        observation, reward, done, info = self.env.step(action)
                        proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                        proccessed_state = np.reshape(proccessed_state, [84, 84])
                        # pop's the item for a given index
                        self.state.pop(0)
                        self.reward.pop(0)
                        #self.value_state.pop(0)
                        self.state.append(proccessed_state)
                        self.reward.append(reward)
                        self.state_buffer.append(self.state)
                        #self.value_state.append(value)
                        self.steps_worker += 1

                        if done:
                            observation = self.env.reset()

                    self.state_buffer.clear()

                if self.steps_worker > 100:
                    coord.request_stop()
                    return


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
