import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import GlobalNetwork


class Worker():
    def __init__(self, cluster, task_id, thread_name, global_network):
        self.task_id = task_id
        self.thread_name = thread_name

        self.local_network = global_network


    def play(self, master_session, coord):

        while not coord.should_stop():

            for i in range(5):
                master_session.run(self.local_network.assign_double)
                sleep(1.0)
                #print(self.thread_name, ": incrementing var current val")

            var = master_session.run(self.local_network.a)
            print(self.thread_name, "Value:", var)

            coord.request_stop()
            return
