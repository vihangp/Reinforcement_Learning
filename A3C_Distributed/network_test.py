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
            self.a = tf.get_variable("a", [1], initializer=tf.constant_initializer(0))
            self.b = tf.get_variable("b", [1], initializer=tf.constant_initializer(1))
            self.c = tf.assign_add(self.a, self.b)