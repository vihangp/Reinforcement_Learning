import multiprocessing
import tensorflow as tf
from time import sleep
from A3C_Distributed.network_test import PolicyValueNetwork

class Worker():
    def __init__(self, master_name, thread_name, global_network):
        self.master_name = master_name
        self.thread_name = thread_name
        self.global_network = global_network

        # with tf.variable_scope(self.thread_name):
        #     self.w_network = PolicyValueNetwork()


    def play(self, local_session, master_session, coord):

        for i in range(5):
            print(self.thread_name, ": incrementing var")
            local_var = master_session.run(self.global_network.var.assign_add(1.0))
            sleep(1.0)



