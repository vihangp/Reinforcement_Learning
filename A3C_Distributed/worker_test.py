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


    def play(self,master_session, coord, supervisor):

        global_step = master_session.run(self.global_network.global_step)

        while not master_session.should_stop() and (not self.num_global_steps or global_step < self.num_global_steps):

            print(self.thread_name,": incrementing var")
            print(global_step)
            _, global_step = master_session.run([self.global_network.var.assign_add(1.0), self.global_network.global_step])
            sleep(1.0)
