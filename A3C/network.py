import tensorflow as tf
import numpy as np


class PolicyValueNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.

    Args:
        number of actions in policy.
    """

    def __init__(self, num_actions):
        with tf.variable_scope("preprocessing"):
            self.observation = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)

            # rgb to grayscale
            self.proc_state = tf.image.rgb_to_grayscale(self.observation)

            # reshape the input
            self.proc_state = tf.image.resize_images(self.proc_state, size=[110, 84],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # this cropping was originally done in the dqn paper because there gpu implementations needed square inputs
            self.proc_state = tf.image.crop_to_bounding_box(self.proc_state, 25, 0, 84, 84)

            # conv modules require float input
            # self.proc_state = tf.to_float(self.proc_state)

        with tf.variable_scope("hidden_layers"):
            # Placeholder for stacked processed states
            self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float16)

            # First conv layer with 16 fliters, 8x8 size of filter, 4 stride of the filter, with ReLu
            self.conv1 = tf.contrib.layers.conv2d(self.state, 16, 8, 4, activation_fn=tf.nn.relu)

            # Second conv layer with 32 filters, 4x4 and stride of 2, with ReLu
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 4, 2, activation_fn=tf.nn.relu)

            # flatten conv output
            self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

            # Fully connected layer with 256 units and ReLu
            self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, 256, activation_fn=tf.nn.relu)

        # Network for policy (state-action function)
        with tf.variable_scope("policy_net"):
            # fully connected layer with number of outputs = number of actions
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, num_actions, activation_fn=None)
            # Soft max over the outputs
            self.state_action = tf.contrib.layers.softmax(self.fc2)
            # squeeze to remove all the 1's from the shape
            self.state_action_out = tf.squeeze(self.state_action)

        # Calculate loss for optimization
        # ...


        with tf.variable_scope("value_net"):
            self.state_value = tf.contrib.layers.fully_connected(self.fc1, 1, activation_fn=None)

            # Calculate loss for optimization
            # ...


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
