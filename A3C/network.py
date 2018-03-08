import tensorflow as tf
import numpy as np


class PolicyValueNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.

    Args:
        number of actions in policy.
    """

    def __init__(self, num_actions):

        # In dqn the frames are
        self.observation = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        # rgb to grayscale
        self.state = tf.image.rgb_to_grayscale(self.observation)
        # reshape the input
        self.state = tf.image.resize_images(self.state, size=[110,84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # this cropping was originally done in the dqn paper because there gpu implementations needed square inputs
        self.proc_state = tf.image.crop_to_bounding_box(self.state, 25, 0, 84, 84)
        # conv modules require float input
        self.proc_state = tf.to_float(self.proc_state)
        # removing 1 from the shape, to make stacking easier.
        self.proc_state = tf.squeeze(self.proc_state)

        #with tf.variable_scope("shared", reuse=True):

        # First conv layer with 16 fliters, 8x8 size of filter, 4 stride of the filter, with ReLu
        self.conv1 = tf.contrib.layers.conv2d(self.proc_state, 16, 8, 4, activation_fn=tf.nn.relu)

        # Second conv layer with 32 filters, 4x4 and stride of 2, with ReLu
        self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 4, 2, activation_fn=tf.nn.relu)

        # flatten conv output
        self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

        # Fully connected layer with 256 units and ReLu
        self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, 256, activation_fn = tf.nn.relu)

        # Network for policy (state-action function)
        with tf.variable_scope("policy_net"):
            self.state_action = tf.contrib.layers.softmax(self.fc1)

        # Calculate loss for optimization
        # ...


        with tf.variable_scope("value_net"):
            self.state_value = tf.contrib.layers.fully_connected(self.fc1, 1,activation_fn = None)

        # Calculate loss for optimization
        # ...