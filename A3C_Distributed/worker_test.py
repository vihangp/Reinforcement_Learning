import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import GlobalNetwork, PolicyValueNetwork


class Worker():
    def __init__(self, cluster, task_id, thread_name, global_network):
        self.task_id = task_id
        self.thread_name = thread_name
        self.global_network = global_network
        self.local_network = PolicyValueNetwork(self.thread_name, task_id)


    def play(self, master_session):

        for i in range(5):
            b = master_session.run(self.local_network.local_var)
            feed_dict = {self.global_network.b: b}
            master_session.run(self.global_network.assign_double, feed_dict)
            sleep(1.0)

        var = master_session.run(self.global_network.a)
        print(self.thread_name, "Value:", var)

