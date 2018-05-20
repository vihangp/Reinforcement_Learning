import tensorflow as tf
import multiprocessing
from time import sleep
from network_test import GlobalNetwork
from worker_test import Worker
import os
import math
import gym

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("node_index", 0, "Index of node for the workers")
tf.app.flags.DEFINE_integer("number_ps", 0, "Number of parameter servers")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("workers_per_node", 1, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS
job_name = FLAGS.job_name
num_ps = FLAGS.number_ps
workers_per_node = FLAGS.workers_per_node

nodes_address = []
ps_list = []
workers_list = []
worker_n = 0
with open('node_list.txt') as nodes:
    for node in nodes:
        node = node.strip('\n')
        nodes_address.append(node)

total_num_nodes = len(nodes_address)
for i, node in enumerate(nodes_address[:num_ps]):
    print(type(node), node)
    print(FLAGS.node_index)
    if node == str(FLAGS.node_index):
        worker_n = i
    ps_list.append("icsnode" + node + ".cluster.net:2222")

for i, node in enumerate(nodes_address[num_ps:]):
    print(type(node), node)
    print(FLAGS.node_index)
    for j in range(workers_per_node):
        if node == str(FLAGS.node_index) and j == FLAGS.task_index:
            worker_n = i * workers_per_node + j
        workers_list.append("icsnode" + node + ".cluster.net:222{}".format(j))

cluster = tf.train.ClusterSpec({
    "worker": workers_list,
    "ps": ps_list
})
# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=worker_n)


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


def parameter_server():

    print("Parameter server: blocking...")
    server.join()


def worker(worker_n):
    game = "Qbert-v0"

    env = gym.make(game)
    num_actions = env.action_space.n
    env.close()

    num_cores = multiprocessing.cpu_count()
    t_max = 20
    print("Num Cores", num_cores)
    gamma = 0.99
    DIR = "/A3C/"
    max_global_time_step = 16000  # 320 * 1000000
    alpha_low = 1e-4
    alpha_high = 1e-2
    alpha_log_rate = 0.4226
    clip_norm = 40.0
    initial_learning_rate = log_uniform(alpha_low, alpha_high, alpha_log_rate)

    MODEL_DIR = 'experiments/exp8'

    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

    global_network = GlobalNetwork(cluster, worker_n, num_actions)

    worker_object = Worker(game, worker_n, t_max, num_actions, global_network, gamma,
                   initial_learning_rate, max_global_time_step, clip_norm)

    hooks = [tf.train.StopAtStepHook(last_step=1000000)]

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(worker_n == 0),
                                           checkpoint_dir=MODEL_DIR, hooks=hooks, save_summaries_steps=1) as master_session:
        print("Worker on Node", FLAGS.node_index, "with Task ID", worker_n)

        worker_object.play(master_session)

if job_name == 'ps':
    print('parameter server')
    parameter_server()
else:
    print('worker server')
    print('Task ID:', worker_n)
    worker(worker_n)