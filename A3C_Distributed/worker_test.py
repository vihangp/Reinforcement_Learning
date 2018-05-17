import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import PolicyValueNetwork

class Worker():
    def __init__(self, task_id, thread_name, global_network):
        self.task_id = task_id
        self.thread_name = thread_name
        self.global_network = global_network

        self.local_network = PolicyValueNetwork(self.thread_name, self.task_id)


    def play(self,master_session, coord):

        for i in range(5):
            print(self.thread_name,": incrementing var")
            master_session.run(self.global_network.var.assign_add(1.0))
            sleep(1.0)
