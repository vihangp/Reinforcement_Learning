import tensorflow as tf
import multiprocessing
from time import sleep
from network_test import GlobalNetwork
from worker_test import Worker
import threading
import os

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("number_ps", 0, "Number of parameter servers")

FLAGS = tf.app.flags.FLAGS
job_name = FLAGS.job_name
num_ps = FLAGS.number_ps

tf.flags.DEFINE_string("model_dir", "experiments/cluster/exp1", "Directory to write Tensorboard summaries and videos to.")
FLAGS = tf.flags.FLAGS
MODEL_DIR = FLAGS.model_dir

CHECKPOINT_DIR = os.path.join(MODEL_DIR,"checkpoints")

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

#writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

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
    print(FLAGS.task_index)
    if node == str(FLAGS.task_index):
        print('Parameter Server index')
        worker_n = i
    print('Parameter Server added ' + str(i))
    ps_list.append("icsnode" + node + ".cluster.net:2222")

for i, node in enumerate(nodes_address[num_ps:]):
    print(type(node), node)
    print(FLAGS.task_index)
    if node == str(FLAGS.task_index):
        print('Worker Server index')
        worker_n = i
    print('Worker Server added ' + str(i))
    workers_list.append("icsnode" + node + ".cluster.net:2222")


cluster = tf.train.ClusterSpec({
    "worker": workers_list,
    "ps": ps_list
})


def parameter_server():
    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=0)
    server.join()
def worker(worker_n):
    server = tf.train.Server(cluster,
                             job_name="worker",
                             task_index=worker_n)

    global_network = GlobalNetwork(cluster, worker_n)
    num_cores = multiprocessing.cpu_count()
    steps = 200

    init_op = tf.global_variables_initializer()

    workers = []
    for i in range(num_cores):
        worker_object = Worker(worker_n, "worker_{}{}".format(FLAGS.task_index, i + 1), global_network, steps)
        workers.append(worker_object)

    # super = tf.train.Supervisor(is_chief=(worker_n == 0),
    #                          init_op=init_op,
    #                          global_step=global_network.global_step
    #                          )
    #
    # with super.managed_session(server.target) as master_session, master_session.as_default():

    with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(worker_n == 0)
                                               ) as master_session:

        coord = tf.train.Coordinator()

        threads = []
        i = 1
        for worker in workers:
            work = lambda worker=worker: worker.play(master_session, coord, super)
            t = threading.Thread(name="worker_{}{}".format(FLAGS.task_index, i + 1), target=work)
            i = i + 1
            threads.append(t)
            t.start()

        coord.join(threads)

    # print("Worker %d: blocking..." % worker_n)
    # super.join()


if job_name == 'ps':
    print('parameter server')
    parameter_server()
else:
    print('worker server')
    worker(worker_n)