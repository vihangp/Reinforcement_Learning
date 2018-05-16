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
    global_network = GlobalNetwork()

    hooks = [tf.train.StopAtStepHook(last_step=1000)]
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           hooks=hooks) as mon_sess:

        while not mon_sess.should_stop():
            var = mon_sess.run(global_network.var.assign_add(1.0))
            print(var)


if job_name == 'ps':
    print('parameter server')
    parameter_server()
else:
    print('worker server')
    worker(worker_n)