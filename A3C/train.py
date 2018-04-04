import gym
from matplotlib import pyplot as plt
import threading
import multiprocessing
from A3C.network import PolicyValueNetwork
from A3C.worker import Worker
import tensorflow as tf
import numpy as np
import os
import math
import itertools


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
env.close()

num_cores = multiprocessing.cpu_count()
t_max = 5
print("Num Cores", num_cores)
gamma = 0.99
DIR = "/A3C/"
max_global_time_step = 320 * 1000000
alpha_low = 1e-4
alpha_high = 1e-2
alpha_log_rate = 0.4226
clip_norm = 40.0
global_counter = itertools.count()

tf.flags.DEFINE_string("model_dir", "experiments/exp1", "Directory to write Tensorboard summaries and videos to.")
FLAGS = tf.flags.FLAGS
MODEL_DIR = FLAGS.model_dir

CHECKPOINT_DIR = os.path.join(MODEL_DIR,"checkpoints")

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))
initial_learning_rate = log_uniform(alpha_low,alpha_high,alpha_log_rate)
#initial_learning_rate = 0.00025

with tf.device("/cpu:0"):
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.variable_scope("global"):
        global_network = PolicyValueNetwork(num_actions, "global")

    global_counter = itertools.count()
    episode_counter = itertools.count()


    workers = []
    for i in range(num_cores):
        worke = Worker(game, "worker_{}".format(i+1), t_max, num_actions, global_network, gamma, writer,
                       initial_learning_rate, max_global_time_step, clip_norm, global_counter, episode_counter)
        workers.append(worke)
    saver = tf.train.Saver(max_to_keep=10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    writer.add_graph(graph=sess.graph)

    threads = []
    i = 1
    for worker in workers:
        work = lambda worker=worker: worker.play(coord, sess, saver, CHECKPOINT_DIR)
        t = threading.Thread(name="Worker_{}".format(i), target=work)
        i = i + 1
        threads.append(t)
        t.start()

    coord.join(threads)


