import numpy as np
import tensorflow as tf


class PolicyGradient(object):
    def __init__(self, n_action, n_feature, learning_rate=0.01, gamma=0.95, output_graph=False):
        self.n_action = n_action
        self.n_feature = n_feature
        self.lr = learning_rate
        self.gamma = gamma
        self.ep_state, self.ep_action, self.ep_reward = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("inputs"):
            self.tf_state = tf.placeholder(tf.float32, [None, 6400], name='observation')
            self.tf_action = tf.placeholder(tf.int32, [None, ], name='action_num')
            self.tf_act_v = tf.placeholder(tf.float32, [None, ], name='action_value')

            layer1 = tf.layers.dense(inputs=self.tf_state, units=100, activation=tf.nn.tanh,
                                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.2),
                                     bias_initializer=tf.constant_initializer(0.1), name='fc1')

            layer2 = tf.layers.dense(inputs=layer1, units=self.n_action, activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.2),
                                     bias_initializer=tf.constant_initializer(0.1), name='fc2')

            self.all_action_prob = tf.nn.softmax(layer2, name='all_prob')

        with tf.name_scope("loss"):
            prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=self.tf_action)
            loss = tf.reduce_mean(prob * self.tf_act_v)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, state):
        prob_act = self.sess.run(self.all_action_prob, feed_dict={self.tf_state: state[np.newaxis, :]})
        action = np.random.choice(range(prob_act.shape[1]), p=prob_act.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_state.append(s)
        self.ep_action.append(a)
        self.ep_reward.append(r)

    def learn(self):
        dis_reward = self._discounted_and_norm_reward()

        self.sess.run(self.train_op,
                      feed_dict={self.tf_state: np.vstack(self.ep_state), self.tf_action: np.array(self.ep_action),
                                 self.tf_act_v: dis_reward})

        self.ep_state, self.ep_action, self.ep_reward = [], [], []
        return dis_reward

    def _discounted_and_norm_reward(self):
        dr = np.zeros_like(self.ep_reward)

        running_add = 0
        for i in reversed(range(0, len(self.ep_reward))):
            running_add = running_add * self.gamma + self.ep_reward[i]
            dr[i] = running_add

        dr -= np.mean(dr)
        dr /= np.std(dr)
        return dr
