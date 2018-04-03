import tensorflow as tf
import numpy as np


class PolicyValueNetwork():
    """
    Used to create the graph of the network. The network will have two heads, one for policy and other for value.

    Args:
        number of actions in policy.
    """

    def __init__(self, num_actions, scope_input):
        with tf.variable_scope("preprocessing"):
            self.observation = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)

            # rgb to grayscale
            self.proc_state = tf.image.rgb_to_grayscale(self.observation)

            # reshape the input
            self.proc_state = tf.image.resize_images(self.proc_state, size=[110, 84],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # this cropping was originally done in the dqn paper because there gpu implementations needed square inputs
            self.proc_state = tf.image.crop_to_bounding_box(self.proc_state, 25, 0, 84, 84)

        with tf.variable_scope("hidden_layers"):
            # Placeholder for stacked processed states
            self.state_u = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)

            self.state = tf.to_float(self.state_u)/255.0

            # First conv layer with 16 fliters, 8x8 size of filter, 4 stride of the filter, with ReLu
            self.conv1 = tf.contrib.layers.conv2d(self.state, 16, 8, 4, activation_fn=tf.nn.relu, trainable=True, weights_initializer = tf.contrib.layers.xavier_initializer())

            # Second conv layer with 32 filters, 4x4 and stride of 2, with ReLu
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 4, 2, activation_fn=tf.nn.relu, trainable=True, weights_initializer = tf.contrib.layers.xavier_initializer())

            # flatten conv output
            self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

            # Fully connected layer with 256 units and ReLu
            self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, 256, activation_fn=tf.nn.relu, trainable=True, weights_initializer = tf.contrib.layers.xavier_initializer())

            # summaries
            #tf.contrib.layers.summarize_activation(self.conv1)
            #tf.contrib.layers.summarize_activation(self.conv2)
            #tf.contrib.layers.summarize_activation(self.fc1)

        # Network for policy (state-action function)
        with tf.variable_scope("policy_net"):
            # fully connected layer with number of outputs = number of actions
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, num_actions, activation_fn=None, trainable=True, weights_initializer = tf.contrib.layers.xavier_initializer())
            # Soft max over the outputs
            self.state_action = tf.contrib.layers.softmax(self.fc2) + 1e-20
            # squeeze to remove all the 1's from the shape
            self.policy = tf.squeeze(self.state_action)

        with tf.variable_scope("value_net"):
            self.value = tf.contrib.layers.fully_connected(self.fc1, 1, activation_fn=None, trainable=True, weights_initializer = tf.contrib.layers.xavier_initializer())
            self.value_transpose = tf.transpose(self.value)

        with tf.variable_scope("loss_calculation"):
            self.advantage = tf.placeholder(shape=[None, None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.actions = tf.squeeze(self.actions)
            self.actions_onehot = tf.squeeze(tf.one_hot(self.actions, num_actions, dtype=tf.float32))
            self.reward = tf.placeholder(shape=[None, None], dtype=tf.float32)

            #logging
            self.episode_reward = tf.Variable(0, name="episode_reward", dtype=tf.float32)

            # policy network loss
            self.entropy = - tf.reduce_sum(self.state_action * tf.log(self.state_action),1)

            # adding a small value to avoid NaN's
            self.log_pi = tf.log(self.state_action)
            self.log_prob_actions = tf.reduce_sum(tf.multiply(self.log_pi , self.actions_onehot),1)
            self.policy_loss = -tf.reduce_sum(self.log_prob_actions * self.advantage + 0.01 * self.entropy)

            # value network loss
            self.value_loss = 0.5 * tf.nn.l2_loss(self.reward - self.value_transpose)

            # total loss
            self.loss = self.value_loss + self.policy_loss

        with tf.variable_scope("optimization"):
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.gradients = self.optimizer.compute_gradients(self.loss)
            self.gradients = [[grad, var] for grad, var in self.gradients if grad is not None]
            self.gradients_apply = self.optimizer.apply_gradients(self.gradients,
                                                                  global_step=tf.train.get_global_step())

        # summary
        #tf.summary.scalar("Total_loss", self.loss)
        #tf.summary.scalar("Entropy", self.entropy)
        #tf.summary.scalar("Policy loss", self.policy_loss)
        #tf.summary.scalar("Value loss", self.value_loss)
        tf.summary.scalar(self.episode_reward.op.name, self.episode_reward)

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "global" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)
