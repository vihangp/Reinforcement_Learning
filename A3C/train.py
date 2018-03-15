import gym
from matplotlib import pyplot as plt
import threading
import multiprocessing
from A3C.network import PolicyValueNetwork
from A3C.worker import Worker
import tensorflow as tf
import numpy as np

game = "Qbert-v0"
# observation/state array shape: (210,160,3)
# every action is performed for a duration of k frames, where k
# is sampled from {2,3,4}
# action space: 6 for Qbert

env = gym.envs.make(game)
num_actions = env.action_space.n
env.close()

num_cores = multiprocessing.cpu_count()
t_max = 5
print("Num Cores", num_cores)
gamma = 0.99

with tf.device("/cpu:0"):
    tf.reset_default_graph()

    with tf.variable_scope("global"):
        global_network = PolicyValueNetwork(num_actions, "global")

    workers = []
    for i in range(num_cores):
        worke = Worker(game, "worker_{}".format(i+1), t_max, num_actions, global_network, gamma)
        workers.append(worke)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="logdir", graph= sess.graph)

        threads = []
        i = 1
        for worker in workers:
            work = lambda worker=worker: worker.play(coord, sess)
            t = threading.Thread(name="Worker_{}".format(i), target=work)
            i = i + 1
            threads.append(t)
            t.start()

        coord.join(threads)


