import tensorflow as tf
import multiprocessing
from time import sleep
from A3C_Distributed.network_test import GlobalNetwork

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("number_ps", 0, "Number of parameter servers")

FLAGS = tf.app.flags.FLAGS
job_name = FLAGS.job_name
num_ps = FLAGS.number_pss

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
    global_network = GlobalNetwork()

    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=0)
    master_session = tf.Session(target=server.target)

    print("Parameter server: waiting for cluster connection...")
    master_session.run(tf.report_uninitialized_variables())
    print("Parameter server: cluster ready!")

    print("Parameter server: initializing variables...")
    master_session.run(tf.global_variables_initializer())
    print("Parameter server: variables initialized")

    for i in range(5):
        val = master_session.run(global_network.var)
        print("Parameter server: var has value %.1f" % val)
        sleep(1.0)

    # sleep(60)
    # val = master_session.run(global_network.var)
    # print("Parameter server: var has value %.1f" % val)
    # sleep(60)
    # val = master_session.run(global_network.var)
    # print("Parameter server: var has value %.1f" % val)

    print("Parameter server: blocking...")
    server.join()


def worker(worker_n):
    global_network = GlobalNetwork()

    server = tf.train.Server(cluster,
                             job_name="worker",
                             task_index=worker_n)
    master_session = tf.Session(target=server.target)

    print("Worker %d: waiting for cluster connection..." % worker_n)
    master_session.run(tf.report_uninitialized_variables())
    print("Worker %d: cluster ready!" % worker_n)

    while master_session.run(tf.report_uninitialized_variables()):
        print("Worker %d: waiting for variable initialization..." % worker_n)
        sleep(1.0)
    print("Worker %d: variables initialized" % worker_n)

    # num_cores = multiprocessing.cpu_count()

    for i in range(5):
        print("Worker %d: incrementing var" % worker_n)
        master_session.run(global_network.var.assign_add(1.0))
        sleep(1.0)

    print("Worker %d: blocking..." % worker_n)
    server.join()


if job_name == 'ps':
    print('parameter server')
    parameter_server()
else:
    print('worker server')
    worker(worker_n)
