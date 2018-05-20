import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import GlobalNetwork
import gym
import operator
import numpy as np


class Worker():
    def __init__(self, game, task_id, t_max, num_actions, global_network, gamma,
                 learning_rate, max_global_time_step, clip_norm):

        self.action = []
        self.value_state = []
        self.state_buffer = []
        self.state = []
        self.reward = []
        self.r_return = []

        self.episode_state = []
        # current steps of the worker
        self.steps_worker = 0
        self.episode_reward = 0

        self.game = game
        self.task_id = task_id
        self.t_max = t_max
        self.num_actions = num_actions
        self.initial_learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.discount = gamma

        self.done = False

        self.max_global_time_step = max_global_time_step

        # Initialise the environment
        self.env = gym.make(self.game)

        self.global_network = global_network


    def play(self, sess):
        learning_rate = self.initial_learning_rate
        global_t = 0
        count = 0

        while not sess.should_stop():

            # create a state buffer from a single state and append it to state buffer
            if self.done or self.steps_worker == 0 or (c_lives != lives):
                # if self.done or self.steps_worker == 0:
                observation = self.env.reset()
                proccessed_state = sess.run([self.global_network.proc_state],
                                            {self.global_network.observation: observation})
                proccessed_state = np.reshape(proccessed_state, [84, 84])
                self.state.clear()
                self.state += 4 * [proccessed_state]
                self.state_buffer.append(self.state[:])
            else:
                # append the last stop state to state buffer
                self.state_buffer.append(self.state[:])

            # interact with the environment for t_max steps or till terminal step
            for t in range(self.t_max):
                # select action
                c_lives = self.env.env.ale.lives()
                action_prob, value = sess.run([self.global_network.policy, self.global_network.value],
                                              {self.global_network.state_u: np.reshape(self.state, [1, 84, 84, 4])})
                action = np.random.choice(np.arange(self.num_actions), p=action_prob)
                # pass action
                observation, reward, self.done, info = self.env.step(action)
                lives = self.env.env.ale.lives()
                # process the new state
                proccessed_state = sess.run([self.global_network.proc_state], {self.global_network.observation: observation})
                proccessed_state = np.reshape(proccessed_state, [84, 84])

                # reward clipping
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1

                # pop's the item for a given index
                self.state.pop(0)
                self.state.append(proccessed_state)
                self.value_state.append(np.reshape(value, [1]))
                self.reward.append(reward)
                self.episode_reward += reward
                self.action.append(action)
                self.state_buffer.append(self.state[:])

                self.steps_worker += 1

                # return the value of the last state
                if self.done or (c_lives != lives):
                    count += 1
                    self.episode_reward = 0
                    self.reward.append(0)
                    break
                elif t == (self.t_max - 1):
                    value = sess.run([self.global_network.value],
                                     {self.global_network.state_u: np.reshape(self.state, [1, 84, 84, 4])})
                    self.reward.append(np.reshape(value, [1]))

            self.r_return = [self.reward[len(self.reward) - 1]]
            num_steps = len(self.state_buffer)

            # number of steps may not always be equal to t_max
            for t in range(num_steps - 1):
                self.r_return.append(self.reward[len(self.reward) - 2 - t] + self.discount * self.r_return[t])

            # removing the value of the last state
            self.r_return.pop(0)
            # reversing the return list to match the indexes of other lists: index order (t+4 -> t)
            self.r_return.reverse()
            # remove the last state from the state buffer, as it will not be used
            self.state_buffer.pop(num_steps - 1)
            num_steps -= 1

            # calculating advantage
            advantage = list(map(operator.sub, self.r_return, self.value_state))
            learning_rate = self.anneal_learning_rate(global_t)

            # popping the value reward from reward buffer
            feed_dict = {
                self.global_network.advantage: np.reshape(advantage, [1, num_steps]),
                self.global_network.actions: np.reshape(self.action, [1, num_steps]),
                self.global_network.state_u: np.reshape(self.state_buffer, [num_steps, 84, 84, 4]),
                self.global_network.reward: np.reshape(self.r_return, [1, num_steps]),
                self.global_network.learning_rate: learning_rate
            }

            # calculating and applying the gradients
            _ = sess.run(
                [self.global_network.gradients_apply], feed_dict)

            self.state_buffer.clear()
            self.reward.clear()
            self.value_state.clear()
            self.action.clear()
            self.r_return.clear()


    def anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
            self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate



