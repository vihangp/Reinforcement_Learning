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
            self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)

            self.state = self.state/255.0

            # First conv layer with 16 fliters, 8x8 size of filter, 4 stride of the filter, with ReLu
            self.conv1 = tf.contrib.layers.conv2d(self.state, 16, 8, 4, activation_fn=tf.nn.relu, trainable=True)

            # Second conv layer with 32 filters, 4x4 and stride of 2, with ReLu
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 4, 2, activation_fn=tf.nn.relu, trainable=True)

            # flatten conv output
            self.conv2_flat = tf.contrib.layers.flatten(self.conv2)

            # Fully connected layer with 256 units and ReLu
            self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, 256, activation_fn=tf.nn.relu, trainable=True)

            # summaries
            #tf.contrib.layers.summarize_activation(self.conv1)
            #tf.contrib.layers.summarize_activation(self.conv2)
            #tf.contrib.layers.summarize_activation(self.fc1)

        # Network for policy (state-action function)
        with tf.variable_scope("policy_net"):
            # fully connected layer with number of outputs = number of actions
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, num_actions, activation_fn=None, trainable=True)
            # Soft max over the outputs
            self.state_action = tf.contrib.layers.softmax(self.fc2) + 1e-8
            # squeeze to remove all the 1's from the shape
            self.policy = tf.squeeze(self.state_action)

        with tf.variable_scope("value_net"):
            self.value = tf.contrib.layers.fully_connected(self.fc1, 1, activation_fn=None, trainable=True)

        with tf.variable_scope("loss_calculation"):
            self.advantage = tf.placeholder(shape=[None, None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
            self.actions = tf.squeeze(self.actions)
            self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
            self.reward = tf.placeholder(shape=[None, None], dtype=tf.float32)
            self.mean_return = tf.reduce_mean(self.reward, name="mean_return")
            self.mean_abs_reward = tf.Variable(0, name= "mean_5_reward", dtype=tf.float32)
            self.mean_100_reward = tf.Variable(0, name= "mean_100_reward", dtype=tf.float32)

            # policy network loss

            self.entropy = - tf.reduce_mean(self.state_action * tf.log(self.state_action))
            # adding a small value to avoid NaN's
            self.log_policy_1 = self.state_action * self.actions_onehot
            self.log_policy_2 = tf.reduce_sum(self.log_policy_1, axis=2, keepdims=False)
            self.log_policy = tf.squeeze(tf.log(self.log_policy_2))

            self.policy_batch_loss = -(self.advantage * self.log_policy)
            self.policy_loss = tf.reduce_mean(self.policy_batch_loss, name="loss")

            # value network loss
            self.value_batch_loss = tf.squared_difference(tf.squeeze(self.value), self.reward)
            self.value_loss = tf.reduce_mean(self.value_batch_loss)

            # total loss
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

        with tf.variable_scope("optimization"):
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.gradients = self.optimizer.compute_gradients(self.loss)
            self.gradients = [[grad, var] for grad, var in self.gradients if grad is not None]
            self.gradients_apply = self.optimizer.apply_gradients(self.gradients,
                                                                  global_step=tf.train.get_global_step())

        # summary
        tf.summary.scalar("Total_loss", self.loss)
        tf.summary.scalar("Entropy", self.entropy)
        tf.summary.scalar("Policy loss", self.policy_loss)
        tf.summary.scalar("Value loss", self.value_loss)
        #tf.summary.scalar(self.mean_return.op.name, self.mean_return)
        #tf.summary.scalar(self.mean_abs_reward.op.name, self.mean_abs_reward)
        tf.summary.scalar(self.mean_100_reward.op.name, self.mean_100_reward)

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "global" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)

            # self.variables_names = [v.name for v in tf.trainable_variables(scope=scope_input)]


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
