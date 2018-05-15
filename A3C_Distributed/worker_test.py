import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import PolicyValueNetwork

class Worker():
    def __init__(self, master_name, thread_name, global_network, local_graph):
        self.master_name = master_name
        self.thread_name = thread_name
        self.global_network = global_network

        with local_graph.as_default():
            with tf.variable_scope(self.thread_name):
                self.local_network = PolicyValueNetwork()


    def play(self, local_session, master_session, coord):

        for i in range(5):
            print(self.thread_name,": incrementing var")
            master_session.run(self.global_network.var.assign_add(1.0))
            sleep(1.0)
