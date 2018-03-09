import tensorflow as tf
import threading
import numpy as np
from A3C.network import PolicyValueNetwork
import gym
from matplotlib import pyplot as plt


class Worker():
    def __init__(self, game, id):

        # current steps of the worker
        self.state = []
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

            if self.steps_worker <4:
                # Select an action
                action = self.env.action_space.sample()
                # Interact with the environment
                observation, reward, done, info = self.env.step(action)
                # process the observation
                proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                # reshape from [1,84,82,1] to [84,84]
                proccessed_state = np.reshape(proccessed_state, [84,84])
                # append the processed state to the end of the list
                self.state.append(proccessed_state)
            else:
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                proccessed_state = np.reshape(proccessed_state, [84, 84])
                # pop's the item from a given index
                self.state.pop(0)
                self.state.append(proccessed_state)

            print(np.shape(self.state))
            self.steps_worker += 1
            print(self.steps_worker, " : ", action)
            if done:
                observation = self.env.reset()

            if self.steps_worker == 100:
                coord.request_stop()
                return
