import gym
from pong_pg import PolicyGradient
import numpy as np
import matplotlib.pyplot as plt

RENDER = False
D = 80 * 80
is_train = True

env = gym.make('Pong-v0')
env.seed(1)
env.unwrapped

print(env.action_space)
print(env.observation_space)

RL = PolicyGradient(n_action=env.action_space.n, n_feature=env.observation_space.shape[0], learning_rate=0.01,
                    gamma=0.99)

if is_train:
    model_file = RL.restore_file
    RL.saver.restore(RL.sess, model_file)


def pre_process(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

for i_episode in range(2000):
    observation = env.reset()
    running_reward = 0
    i = 0
    pre_state = None
    acc_reward = 0
    while True:
        if RENDER:
            env.render()

        cur_state = pre_process(observation)
        x = cur_state - pre_state if pre_state is not None else np.zeros(D)
        pre_state = cur_state

        action = RL.choose_action(x)

        observation_next, reward, done, info = env.step(action)
        RL.store_transition(x, action, reward)
        acc_reward += reward

        i += 1
        if done:
            episode_reward_sum = sum(RL.ep_reward)

            if 'running_reward' not in globals():
                running_reward = episode_reward_sum
            else:
                running_reward = running_reward * 0.99 + episode_reward_sum * 0.01

            if running_reward > -100:
                RENDER = True

            print("i=%d" % i)
            print("episode:", i_episode, "  reward:", int(running_reward), "  acc_reward:", acc_reward)

            action_value = RL.learn()

            if is_train:
                print("learn_value=", sum(action_value))
                print("****************************")
                RL.saver.save(RL.sess, 'ckpt/pong_policy_gradient/pong_pg.ckpt')

            if i_episode == 100:
                plt.plot(action_value)
                plt.xlabel("episode_steps")
                plt.ylabel("normalized state_action value")
                plt.show()
            break

        observation = observation_next
