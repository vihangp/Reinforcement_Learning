import tensorflow as tf
import multiprocessing

class GlobalNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.
    Args:
        number of actions in policy.
    """

    def __init__(self):

        with tf.device("/job:ps/task:0"):
            self.global_step = tf.train.get_or_create_global_step()
            # self.a = tf.placeholder(dtype=tf.float32, shape=[], name="a")
            # self.b = tf.get_variable("b", [1], initializer=tf.constant_initializer(0))
            # self.c = self.b + self.a
            # self.b = self.c

            self.a = tf.Variable([1.0], dtype=tf.float32)
            self.assign_double = tf.assign(self.a, 1+ self.a)


class PolicyValueNetwork():
    def __init__(self, thread_name, task_id):

        with tf.device("/job:worker/task:{}".format(task_id)):

            with tf.variable_scope(thread_name):

                self.local_var = tf.Variable(50, name='var')


