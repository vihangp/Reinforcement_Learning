import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import PolicyValueNetwork, GlobalNetwork


class Worker():
    def __init__(self, task_id, thread_name, graph):
        self.task_id = task_id
        self.thread_name = thread_name

        self.local_network = GlobalNetwork(self.thread_name, self.task_id)


    def play(self, master_session, coord):

        while not coord.should_stop():

            for i in range(5):
                master_session.run(self.local_network.assign_double)
                sleep(1.0)
                print(self.thread_name, ": incrementing var current val")

            coord.request_stop()
            return
