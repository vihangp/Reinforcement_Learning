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

    # for i in range(5):
    #     val = master_session.run(global_network.var)
    #     print("Parameter server: var has value %.1f" % val)
    #     sleep(1.0)

    sleep(60)
    val = master_session.run(global_network.var)
    print("Parameter server: var has value %.1f" % val)
    sleep(60)
    val = master_session.run(global_network.var)
    print("Parameter server: var has value %.1f" % val)

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

    num_cores = multiprocessing.cpu_count()


    local_graph = tf.Graph()

    # A session initialised under this graph will be use this graph as default
    # As we already have given the master_session the global_network, we dont have to worry about it anymore
    workers = []
    for i in range(num_cores):
        worker_object = Worker(worker_n, "worker_{}{}".format(FLAGS.task_index, i + 1), global_network, local_graph)
        workers.append(worker_object)

    local_session = tf.Session(graph=local_graph)
    local_session.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()

    threads = []
    i = 1
    for worker in workers:
        work = lambda worker=worker: worker.play(local_session, master_session, coord)
        t = threading.Thread(name="worker_{}{}".format(FLAGS.task_index, i + 1), target=work)
        i = i + 1
        threads.append(t)
        t.start()

    coord.join(threads)

    print("Worker %d: blocking..." % worker_n)
    server.join()


if job_name == 'ps':
    print('parameter server')
    parameter_server()
else:
    print('worker server')
    worker(worker_n)