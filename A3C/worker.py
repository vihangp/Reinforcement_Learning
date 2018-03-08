import tensorflow as tf
import threading
import numpy as np
from A3C.network import PolicyValueNetwork
import gym
from matplotlib import pyplot as plt


class Worker():
    def __init__(self, game, id):

        # current steps of the worker
        self.steps_worker = 0
        self.game = game
        self.id = id

        # Initialise the environment
        self.env = gym.envs.make(self.game)
        # Size of the action space
        self.num_actions = self.env.action_space.n

        # Worker network
        with tf.variable_scope(self.id):
            self.w_network = PolicyValueNetwork(self.num_actions)

    def play(self, coord, sess):

        observation = self.env.reset()

        while not coord.should_stop():

            self.steps_worker += 1
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
            print(np.shape(proccessed_state))
            print(self.steps_worker, " : ", action)


            if done:
                observation = self.env.reset()

            if self.steps_worker == 100:
                coord.request_stop()
                return
