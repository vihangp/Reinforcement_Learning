import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import GlobalNetwork, PolicyValueNetwork


class Worker():
    def __init__(self, cluster, task_id, thread_name, global_network):
        self.task_id = task_id
        self.thread_name = thread_name
        self.global_network = global_network


    def play(self, master_session):

        for i in range(5):
            master_session.run(self.global_network.assign_double)
            sleep(1.0)

        var = master_session.run(self.global_network.a)
        print(self.thread_name, "Value:", var)

