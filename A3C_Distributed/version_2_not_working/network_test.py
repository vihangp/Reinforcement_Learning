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
            self.var = tf.Variable(0.0, name='var')


class PolicyValueNetwork():
    def __init__(self):
        with tf.variable_scope("preprocessing"):
            self.local_var = tf.Variable(50, name='var')