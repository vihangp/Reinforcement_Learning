import tensorflow as tf
import multiprocessing
from time import sleep
from network_test import GlobalNetwork
from worker_test import Worker
import threading

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("number_ps", 0, "Number of parameter servers")

FLAGS = tf.app.flags.FLAGS
job_name = FLAGS.job_name
num_ps = FLAGS.number_ps

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
    ps_list.append("isnode" + node + ".cluster.net:2222")

for i, node in enumerate(nodes_address[num_ps:]):
    print(type(node), node)
    print(FLAGS.task_index)
    if node == str(FLAGS.task_index):
        print('Worker Server index')
        worker_n = i
    print('Worker Server added ' + str(i))
    workers_list.append("isnode" + node + ".cluster.net:2222")

cluster = tf.train.ClusterSpec({
    "worker": workers_list,
    "ps": ps_list
})
# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=worker_n)


def parameter_server():

    print("Parameter server: blocking...")
    server.join()


def worker(worker_n):

    global_network = GlobalNetwork(cluster, worker_n)

    worker_object = Worker(cluster, worker_n, "worker_{}".format(FLAGS.task_index), global_network)


    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(worker_n == 0)) as master_session:

        while not master_session.should_stop():

            worker_object.play(master_session)

if job_name == 'ps':
    print('parameter server')
    parameter_server()
else:
    print('worker server')
    worker(worker_n)