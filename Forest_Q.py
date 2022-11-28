import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from gym.envs.toy_text import frozen_lake
import time
import mdptoolbox.example

exec_time = []
reward = []
iter_list = []
Q_table = []

for states in [100, 200, 300]:
    P, R = mdptoolbox.example.forest(S=states)
    mean_value = []
    policy = []
    iters = []
    time_array = []
    gamma_array = []
    best_policy = None
    best_score = 0
    for i in range(0, 10):
        pi = mdptoolbox.mdp.QLearning(P, R, (i+0.5)/10)
        pi.run()
        gamma_array.append((i+0.5)/10)
        mean_value.append(np.mean(pi.V))
        if np.mean(pi.V) > best_score:
            best_policy = np.array(pi.policy)
            best_score = np.mean(pi.V)
        policy.append(pi.policy)
        time_array.append(pi.time)
    exec_time.append(time_array)
    reward.append(mean_value)
    iter_list.append(iters)
    best_policy = best_policy.astype(str)
    best_policy[best_policy == '0'] = 'W'
    best_policy[best_policy == '1'] = 'C'
    print(best_policy)

plt.plot(gamma_array, exec_time[0], label='State - 100')
plt.plot(gamma_array, exec_time[1], label='State - 200')
plt.plot(gamma_array, exec_time[2], label='State - 300')
plt.xlabel('Gamma')
plt.title('Q-Learning - Gamma vs Execution Time')
plt.ylabel('Execution Time (s)')
plt.grid()
plt.legend()
plt.savefig("images/forest/Q_ExecutionTime.png")
#plt.show()
plt.clf()

plt.plot(gamma_array, reward[0], label='State - 100')
plt.plot(gamma_array, reward[1], label='State - 200')
plt.plot(gamma_array, reward[2], label='State - 300')
plt.xlabel('Gamma')
plt.ylabel('Average Rewards')
plt.title('Q-Learning - Gamma vs Reward')
plt.grid()
plt.legend()
plt.savefig("images/forest/Q_AverageRewards.png")
#plt.show()
plt.clf()

