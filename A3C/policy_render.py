import gym
import multiprocessing
from A3C.network import PolicyValueNetwork
import tensorflow as tf
import numpy as np
import os
import math
import time


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)

game = "Qbert-v0"
#game = "Pong-v0"
#game = "Breakout-v0"
#game = "CartPole-v0"
# observation/state array shape: (210,160,3)
# every action is performed for a duration of k frames, where k
# is sampled from {2,3,4}
# action space: 6 for Qbert
# 0,1,2,3,4,5 : possible actions

env = gym.make(game)
num_actions = env.action_space.n

action = []
state = []
done = False

num_cores = multiprocessing.cpu_count()
t_max = 5
print("Num Cores", num_cores)
gamma = 0.99
DIR = "/A3C/"
steps = 100 #320 * 1000000
episodes = 5

tf.flags.DEFINE_string("model_dir", "render/exp6", "Directory to write Tensorboard summaries and videos to.")
FLAGS = tf.flags.FLAGS
MODEL_DIR = FLAGS.model_dir

CHECKPOINT_DIR = os.path.join(MODEL_DIR,"checkpoints_1")

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

print(CHECKPOINT_DIR)

with tf.device("/cpu:0"):
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.variable_scope("global"):
        global_network = PolicyValueNetwork(num_actions, "global")
    saver = tf.train.Saver(max_to_keep=10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    while episodes != 0:
        steps = 0
        if  steps == 0:
            # if self.done or self.steps_worker == 0:
            observation = env.reset()
            proccessed_state = sess.run([global_network.proc_state],
                                        {global_network.observation: observation})
            proccessed_state = np.reshape(proccessed_state, [84, 84])
            state.clear()
            state += 4 * [proccessed_state]
            done = False

        # interact with the environment for t_max steps or till terminal step
        while done!= True:
            # select action
            env.render()
            time.sleep(0.01)
            c_lives = env.env.ale.lives()
            action_prob = sess.run([global_network.policy],
                                          {global_network.state_u: np.reshape(state, [1, 84, 84, 4])})

            #s_action = np.random.choice(np.arange(num_actions), p=action_prob)
            action = np.argmax(action_prob)
            #action = env.action_space.sample()
            # pass action
            observation, reward, done, info = env.step(action)
            lives = env.env.ale.lives()
            # process the new state
            proccessed_state = sess.run([global_network.proc_state], {global_network.observation: observation})
            proccessed_state = np.reshape(proccessed_state, [84, 84])

            # reward clipping
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1

            # pop's the item for a given index
            state.pop(0)
            state.append(proccessed_state)
            steps += 1

            # return the value of the last state
            if done or (c_lives != lives):
                print("Episode Done")
                break

        episodes -= 1
        print(episodes)