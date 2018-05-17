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
        print("Initializing:", self.thread_name)


    def play(self,master_session, coord):
        print("I am Survivor:",self.thread_name)

        while not coord.should_stop():

            _, global_step = master_session.run([self.global_network.var.assign_add(1.0), self.global_network.global_step.assign_add(1.0)])
            sleep(1.0)
            print(self.thread_name, ": incrementing var")
            print(global_step)

            if global_step > self.num_global_steps:
                coord.request_stop()
                return
