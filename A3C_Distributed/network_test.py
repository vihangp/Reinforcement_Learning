import tensorflow as tf
import multiprocessing

class GlobalNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.
    Args:
        number of actions in policy.
    """

    def __init__(self, cluster, task_id):

        worker_device = "/job:worker/task:{}".format(task_id)
        with tf.device(tf.train.replica_device_setter( worker_device=worker_device, cluster=cluster)):

            self.global_step = tf.train.get_or_create_global_step()
            self.a = tf.get_variable("a", [1], initializer=tf.constant_initializer(0))
            self.b = tf.get_variable("b", [1], initializer=tf.constant_initializer(1))
            self.c = tf.assign_add(self.a, self.b)

class PolicyValueNetwork():
    def __init__(self, thread_name, task_id):
        worker_device = "/job:worker/task:{}".format(task_id)

        with tf.device("/job:worker/task:{}".format(task_id)):
            with tf.variable_scope("thread_name"):

                self.local_var = tf.Variable(50, name='var')