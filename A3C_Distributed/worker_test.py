import multiprocessing
import tensorflow as tf
from time import sleep

class Worker():
    def __init__(self, master_name, thread_name, global_network):
        self.master_name = master_name
        self.thread_name = thread_name
        self.global_network = global_network


    def play(self, local_session, master_session, coord):

        for i in range(5):
            print(self.thread_name,": incrementing var")
            master_session.run(self.global_network.var.assign_add(1.0))
            sleep(1.0)

