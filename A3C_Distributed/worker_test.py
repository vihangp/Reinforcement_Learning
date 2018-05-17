import multiprocessing
import tensorflow as tf
from time import sleep
from network_test import PolicyValueNetwork

class Worker():
    def __init__(self, task_id, thread_name, global_network, steps):
        self.task_id = task_id
        self.thread_name = thread_name
        self.global_network = global_network
        self.num_global_steps = steps

        self.local_network = PolicyValueNetwork(self.thread_name, self.task_id)
        print("Initializing:", self.thread_name, " Task Id", self.task_id)
        self.global_step = 0

    def play(self,master_session, coord):

        while not coord.should_stop():

            #_, global_step = master_session.run([self.global_network.var.assign_add(1.0), self.global_network.global_step.assign_add(1.0)])

            for i in range(10):
                var_value = master_session.run(self.global_network.c)
                sleep(1.0)
                print(self.thread_name, ": incrementing var, current val", var_value)

            # if self.global_step > self.num_global_steps:
            #     coord.request_stop()
            #     return

            coord.request_stop()
            return
