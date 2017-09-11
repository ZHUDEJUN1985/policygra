import numpy as np
import gym
import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

# LR_A = 0.001  # learning rate for actor
# LR_C = 0.002  # learning rate for critic
# GAMMA = 0.95  # reward discount
# TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
D = 80 * 80

RENDER = False
is_train = True


class DDPG(object):
    def __init__(self, a_dim, s_dim, lr_a, lr_c, gamma, tau):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 2), dtype=np.float32)
        self.pointer = 0
        self.lr_actor = lr_a
        self.lr_critic = lr_c
        self.gamma = gamma
        self.tau = tau
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_critic).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_actor).minimize(a_loss, var_list=self.ae_params)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore_file = tf.train.latest_checkpoint('ckpt/pong_deterministic_pg/')

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + 1]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, 1, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


env = gym.make('Pong-v0')
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
# a_bound = env.action_space.high
print(s_dim)
print(a_dim)

ddpg = DDPG(a_dim, D, 0.01, 0.02, 0.98, 0.01)

if is_train:
    model_file = ddpg.restore_file
    ddpg.saver.restore(ddpg.sess, model_file)


def pre_process(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


var = 3  # control exploration
for i in range(3000):
    observation = env.reset()
    ep_reward = 0
    pre_state = None
    for j in range(200):
        if RENDER:
            env.render()

        # Add exploration noise
        cur_state = pre_process(observation)
        x = cur_state - pre_state if pre_state is not None else np.zeros(D)
        pre_state = cur_state

        act = ddpg.choose_action(x)
        act = np.clip(np.random.normal(act, var), -2, 3)  # add randomness to action selection for exploration
        action_step = int(act)
        action_step += 2
        s_, r, done, info = env.step(action_step)

        x_ = pre_process(s_)
        ddpg.store_transition(x, action_step, r, x_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995  # decay the action randomness
            ddpg.learn()

        observation = s_
        ep_reward += r

        if j == 10:
            print("act=%d, r=%.2f" % (action_step, r))
        elif j == 30:
            print("act=%d, r=%.2f" % (action_step, r))
        elif j == 50:
            print("act=%d, r=%.2f" % (action_step, r))
        elif j == 100:
            print("act=%d, r=%.2f" % (action_step, r))
        elif j == 150:
            print("act=%d, r=%.2f" % (action_step, r))

        if j == 199:
            print('Episode:', i, ' Reward: %.2f' % ep_reward, 'Explore: %.2f' % var)
            print('**************************************')
            ddpg.saver.save(ddpg.sess, 'ckpt/pong_deterministic_pg/ddpg.ckpt')
            if ep_reward > -30:
                RENDER = True
            break
