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
    def __init__(self, game, id, t_max, num_actions, global_network, gamma):

        self.action = []
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
        self.done = []

        # Initialise the environment
        self.env = gym.envs.make(self.game)
        # Size of the action space
        self.num_actions = num_actions

        # Worker network
        with tf.variable_scope(self.id):
            self.w_network = PolicyValueNetwork(self.num_actions, self.id)

        # Cannot change uninitialised graph concurrently.
        self.copy_network = copy_network("global", self.id)

        self.discount = gamma

    def play(self, coord, sess):
        with sess.as_default(), sess.graph.as_default():

            # observation = self.env.reset()

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

            while not coord.should_stop():

                sess.run(self.copy_network)
                t = 0

                if self.done or self.steps_worker == 0:
                    print(threading.current_thread().name, ": starting a new episode")
                    observation = self.env.reset()
                    proccessed_state = sess.run([self.w_network.proc_state],
                                                {self.w_network.observation: observation})
                    proccessed_state = np.reshape(proccessed_state, [84, 84])
                    self.state.clear()
                    self.state += 4 * [proccessed_state]
                    self.state_buffer.append(self.state)
                else:
                    self.state_buffer.append(self.state)

                for t in range(self.t_max):
                    action_prob, value = sess.run([self.w_network.policy, self.w_network.value],
                                                  {self.w_network.state: np.reshape(self.state, [1, 84, 84, 4])})
                    action = np.random.choice(np.arange(self.num_actions), p=action_prob)
                    observation, reward, self.done, info = self.env.step(action)
                    proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                    proccessed_state = np.reshape(proccessed_state, [84, 84])
                    # pop's the item for a given index
                    self.state.pop(0)
                    self.state.append(proccessed_state)
                    self.value_state.append(np.reshape(value, [1]))
                    self.reward.append(reward)
                    self.action.append(action)
                    self.state_buffer.append(self.state)
                    self.steps_worker += 1

                    if t == (self.t_max - 1):
                        value = sess.run([self.w_network.value],
                                         {self.w_network.state: np.reshape(self.state, [1, 84, 84, 4])})
                        self.reward.append(np.reshape(value, [1]))

                    if self.done:
                        t_state = 0
                        break

                #reward_array = np.array(self.reward)
                #print(reward_array)
                #print(reward_array[len(self.reward)-1])
                # calculate the advantage
                #return_tmax = np.sum(reward_array * self.discount)
                #advantage = return_tmax - self.value_state[0]

                #if threading.current_thread().name == "Worker_1":

                r_return = [self.reward[len(self.reward) - 1]]

                for t in range(self.t_max):
                    r_return.append(self.reward[len(self.reward) - 2 - t] + self.discount * r_return[0])

                self.state_buffer.clear()
                self.reward.clear()
                self.value_state.clear()
                self.action.clear()

                if self.steps_worker > 400:
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
