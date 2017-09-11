import gym
from pg import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -200
RENDER = False
is_train = False

env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)

RL = PolicyGradient(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.01,
                    reward_decay=0.98)

if not is_train:
    model_file = RL.restore_file
    RL.saver.restore(RL.sess, model_file)

max_reward = -200
for i_episode in range(1000):
    observation = env.reset()
    running_reward = 0
    i = 0
    while True:
        if RENDER:
            env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        i += 1
        if i % 1000 == 0:
            print("i=%d, action=%d" % (i, action))

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if is_train and running_reward > max_reward:
                max_reward = running_reward
                RL.saver.save(RL.sess, 'ckpt/car_pole/car_pole.ckpt')

            if i_episode == 30:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
