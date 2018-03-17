import tensorflow as tf
import threading
import numpy as np
from A3C.network import PolicyValueNetwork
import gym
import operator
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

def make_train_op(local_estimator, global_estimator):

  local_grads, _ = zip(*local_estimator.gradients)
  # Clip gradients
  local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
  _, global_vars = zip(*global_estimator.gradients)
  local_global_grads_and_vars = list(zip(local_grads, global_vars))
  return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.train.get_global_step())


class Worker():
    def __init__(self, game, id, t_max, num_actions, global_network, gamma, summary_writer):

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
        self.r_return = []
        self.writer = summary_writer
        self.global_step = tf.train.get_global_step()

        # Initialise the environment
        self.env = gym.envs.make(self.game)
        # Size of the action space
        self.num_actions = num_actions

        # Worker network
        with tf.variable_scope(self.id):
            self.w_network = PolicyValueNetwork(self.num_actions, self.id)

        # Cannot change uninitialised graph concurrently.
        self.copy_network = copy_network("global", self.id)

        self.grad_apply = make_train_op(self.w_network, self.global_network)

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

                # create a state buffer from a single state and append it to state buffer
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
                    # append the last stop state to state buffer
                    self.state_buffer.append(self.state)

                # interact with the environment for t_max steps or till terminal step
                for t in range(self.t_max):
                    # select action
                    action_prob, value = sess.run([self.w_network.policy, self.w_network.value],
                                                  {self.w_network.state: np.reshape(self.state, [1, 84, 84, 4])})
                    action = np.random.choice(np.arange(self.num_actions), p=action_prob)
                    # pass action
                    observation, reward, self.done, info = self.env.step(action)
                    # process the new state
                    proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
                    proccessed_state = np.reshape(proccessed_state, [84, 84])

                    # if threading.current_thread().name == "Worker_1":
                    #    print(action)

                    # pop's the item for a given index
                    self.state.pop(0)
                    self.state.append(proccessed_state)

                    self.value_state.append(np.reshape(value, [1]))
                    self.reward.append(reward)
                    self.action.append(action)

                    self.state_buffer.append(self.state)
                    self.steps_worker += 1

                    # give return the value of the last state
                    if t == (self.t_max - 1):
                        value = sess.run([self.w_network.value],
                                         {self.w_network.state: np.reshape(self.state, [1, 84, 84, 4])})
                        self.reward.append(np.reshape(value, [1]))

                    if self.done:
                        break

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
                advantage = list(map(operator.sub,self.r_return, self.value_state ))

                feed_dict = {
                    self.w_network.advantage: np.reshape(advantage, [1, num_steps]),
                    self.w_network.actions: np.reshape(self.action, [1, num_steps]),
                    self.w_network.state: np.reshape(self.state_buffer, [num_steps, 84, 84, 4]),
                    self.w_network.reward: np.reshape(self.r_return, [1, num_steps])
                }

                _, summaries, global_step= sess.run([self.grad_apply,
                           self.w_network.summaries,
                           self.global_step], feed_dict)

                self.writer.add_summary(summaries, global_step)
                self.writer.flush()

                self.state_buffer.clear()
                self.reward.clear()
                self.value_state.clear()
                self.action.clear()
                self.r_return.clear()

                if self.steps_worker > 2000:
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
