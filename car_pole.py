import gym
from pg import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000
RENDER = False

env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.02,
                    reward_decay=0.995)

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

        if i % 200 == 0:
            print("i=%d" % i)
            print(action)
        i += 1
        if done:
            ep_rs_sum = sum(RL.ep_rs)
            print('*******************************')
            print(done)
            print(action)
            print('*******************************')

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 30:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
