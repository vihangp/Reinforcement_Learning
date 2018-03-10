import tensorflow as tf
import threading
import numpy as np
from A3C.network import PolicyValueNetwork
import gym
from matplotlib import pyplot as plt


class Worker():
    def __init__(self, game, id, t_max, num_actions, global_network):

        # current steps of the worker
        self.state = []
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
            self.w_network = PolicyValueNetwork(self.num_actions)



    def play(self, coord, sess):

        observation = self.env.reset()



        while not coord.should_stop():

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
            else:
                action_prob = sess.run([self.w_network.state_action_out],
                                       {self.w_network.state: np.reshape(self.state, [1, 84, 84, 4])})
                action = np.argmax(action_prob, axis=1)
                print(action)
                observation, reward, done, info = self.env.step(action)
                proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                proccessed_state = np.reshape(proccessed_state, [84, 84])
                # pop's the item for a given index
                self.state.pop(0)
                self.state.append(proccessed_state)

            self.steps_worker += 1
            if done:
                observation = self.env.reset()

            if self.steps_worker == 1000:
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
