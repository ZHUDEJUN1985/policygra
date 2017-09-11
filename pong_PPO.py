import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
A_DIM = 1
D = 80 * 80
is_train = False
RENDER = False
METHOD = [dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
          dict(name='clip', epsilon=0.2)][1]


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, 6400], 'state')

        # critic
        with tf.name_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.name_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        with tf.name_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.name_scope('loss'):
            with tf.name_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.stop_gradient(tf.distributions.kl_divergence(oldpi, pi))
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:  # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))

        with tf.name_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore_file = tf.train.latest_checkpoint('ckpt/pong_ppo/')

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            kl = 0.0
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # some time explode, this is my method
        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.name_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


env = gym.make('Pong-v0')
env.seed(1)
env.unwrapped


def pre_process(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


ppo = PPO()
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
print(s_dim)
print(a_dim)

all_ep_r = []
for ep in range(3000):
    observation = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_reward = 0
    pre_state = None
    for t in range(200):  # in one episode
        # if RENDER:
        #    env.render()

        cur_state = pre_process(observation)
        x = cur_state - pre_state if pre_state is not None else np.zeros(D)
        pre_state = cur_state

        a = ppo.choose_action(x)
        s_, r, done, _ = env.step(int(a))
        buffer_s.append(x)
        buffer_a.append(a)
        buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
        observation = s_
        ep_reward += r

        # update ppo
        if (t + 1) % BATCH == 0 or t == 199:
            s_pre = pre_process(s_)
            x_ = s_pre - cur_state
            v_s_ = ppo.get_v(x_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
            ppo.saver.save(ppo.sess, 'ckpt/pong_ppo/ppo.ckpt')
            print("t=%d , reward=%.2f" % (t, ep_reward))

    if ep == 0:
        all_ep_r.append(ep_reward)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_reward * 0.1)

    # if ep_reward > -30:
    #    RENDER = True

    print('Ep: %i' % ep, "|Ep_r: %i" % ep_reward,
          ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '')
    print('**********************************')

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
