import tensorflow as tf
import threading
import numpy as np
from A3C.network import PolicyValueNetwork
import gym
import operator
import itertools


def copy_network(from_scope, to_scope):
    global_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    local_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    copy_val = []
    for g, w in zip(local_val, global_val):
        op = g.assign(w)
        copy_val.append(op)

    return copy_val


# Include learning rate
def make_train_op(local_estimator, global_estimator, clip_norm):
    local_grads, _ = zip(*local_estimator.gradients)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, clip_norm)
    _, global_vars = zip(*global_estimator.gradients)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
                                                      global_step=tf.train.get_global_step())


class Worker():
    def __init__(self, game, id, t_max, num_actions, global_network, gamma, summary_writer,
                 learning_rate, max_global_time_step, clip_norm, global_counter, episode_counter):

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
        self.id = id
        self.t_max = t_max
        self.num_actions = num_actions
        self.initial_learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.discount = gamma

        self.done = False
        self.writer = summary_writer

        self.max_global_time_step = max_global_time_step
        self.global_step = tf.train.get_global_step()
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.episode_counter = episode_counter

        # Initialise the environment
        self.env = gym.make(self.game)

        self.global_network = global_network
        # Worker network
        with tf.variable_scope(self.id):
            self.w_network = PolicyValueNetwork(self.num_actions, self.id)

        # Cannot change uninitialised graph concurrently.
        self.copy_network = copy_network("global", self.id)

        self.grad_apply = make_train_op(self.w_network, self.global_network,
                                        self.clip_norm)  # include learning rate as one of the input

    def play(self, coord, sess, saver, CHECKPOINT_DIR):
        with sess.as_default(), sess.graph.as_default():
            learning_rate = self.initial_learning_rate
            global_t = 0
            episode_count = 0
            count = 0

            while not coord.should_stop():

                sess.run(self.copy_network)
                t = 0
                # if self.steps_worker < 100000:
                #     lives = 4
                # create a state buffer from a single state and append it to state buffer
                if self.done or self.steps_worker == 0 or (c_lives != lives):
                    # if self.done or self.steps_worker == 0:
                    observation = self.env.reset()
                    proccessed_state = sess.run([self.w_network.proc_state],
                                                {self.w_network.observation: observation})
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
                    if threading.current_thread().name == "Worker_2":
                        self.env.render()
                    c_lives = self.env.env.ale.lives()
                    action_prob, value = sess.run([self.w_network.policy, self.w_network.value],
                                                  {self.w_network.state_u: np.reshape(self.state, [1, 84, 84, 4])})
                    action = np.random.choice(np.arange(self.num_actions), p=action_prob)
                    # pass action
                    observation, reward, self.done, info = self.env.step(action)
                    lives = self.env.env.ale.lives()
                    # process the new state
                    proccessed_state = sess.run([self.w_network.proc_state], {self.w_network.observation: observation})
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
                    local_t = next(self.local_counter)
                    global_t = next(self.global_counter)

                    if local_t % 100 == 0:
                        tf.logging.info("{}: local Step {}, global step {}".format(self.id, local_t, global_t))

                    # return the value of the last state
                    if self.done or (c_lives != lives):
                        episode_count = next(self.episode_counter)
                        count +=1
                        if threading.current_thread().name == "Worker_1":
                            summaries, global_step = sess.run(
                                [self.w_network.summaries,
                                 self.global_step], feed_dict={self.w_network.episode_reward: self.episode_reward}
                            )
                            self.writer.add_summary(summaries, global_step)
                            self.writer.flush()
                            if count % 5 == 0:
                                print("Global Episode Count:",episode_count)
                                print("Global Steps",global_t)
                        self.episode_reward = 0
                        self.reward.append(0)
                        break
                    elif t == (self.t_max - 1):
                        value = sess.run([self.w_network.value],
                                         {self.w_network.state_u: np.reshape(self.state, [1, 84, 84, 4])})
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
                    self.w_network.advantage: np.reshape(advantage, [1, num_steps]),
                    self.w_network.actions: np.reshape(self.action, [1, num_steps]),
                    self.w_network.state_u: np.reshape(self.state_buffer, [num_steps, 84, 84, 4]),
                    self.w_network.reward: np.reshape(self.r_return, [1, num_steps]),
                    self.w_network.learning_rate: learning_rate,
                    self.global_network.learning_rate: learning_rate
                }

                # calculating and applying the gradients
                _ = sess.run(
                     [self.grad_apply], feed_dict)

                self.state_buffer.clear()
                self.reward.clear()
                self.value_state.clear()
                self.action.clear()
                self.r_return.clear()

                if global_t > self.max_global_time_step:
                    coord.request_stop()
                    return

                if threading.current_thread().name == "Worker_1":
                    if count % 50 == 0 and count != 0:
                            saver.save(sess, CHECKPOINT_DIR + '/model-' + str(episode_count) + '.cptk')
                            print("Saved Model")

    def anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
        self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


        # To Do
        # 5) Check Return - Check one more time
        # 12) Action repeat to calculate initial 4 frames
        # 13) config file for flags or inputs
        # 14) Make use of functions
        # 17) Make sure that experiments have config file which can be imported

