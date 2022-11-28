import numpy as np
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt
from gym.envs.toy_text import frozen_lake


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    start_time = time.time()
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done or time.time()-start_time > .1:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    max_iterations = 2000
    eps = 1e-20
    k=0
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k = i+1
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v, k


def show_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.savefig('images/VI/'+title + str('.png'))
    plt.close()
    #plt.show()


def colors():
    return {
        b'S': 'green',
        b'F': 'skyblue',
        b'H': 'black',
        b'G': 'gold',
    }


def directions():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }


if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    np.random.seed(42)

    exec_time_lst = []
    score_lst_lst = []
    conv_iter_lst = []

    for size in [4, 8, 10, 15, 20]:
        exec_time = []
        score_lst = []
        conv_iter = []
        gamma_list = [0.2, 0.4, 0.8, 0.9, 0.99]

        custom_map = frozen_lake.generate_random_map(size=size, p=0.9)
        option = str(size) + "*" + str(size)

        for i, gamma in enumerate(gamma_list):
            env = gym.make(env_name, desc=custom_map)
            env.reset()
            #env.render()
            env = env.unwrapped
            desc = env.unwrapped.desc
            start_time = time.time()
            best_val, k = value_iteration(env)
            policy = extract_policy(best_val, gamma)
            scores = evaluate_policy(env, policy, gamma, n=1000)
            exec_time.append(round(time.time() - start_time, 2))
            score_lst.append(scores)
            conv_iter.append(k)
            print(option, ' : ', gamma, ' : ', k, ' : ', round(scores, 3))
            #print('Average scores = ', scores)
            show_policy_map(
                'Frozen Lake  ' + option + ' VI Policy Map - Iteration ' + str(k) + ' Gamma -  ' + str(gamma),
                policy.reshape(size, size), desc, colors(), directions())
        exec_time_lst.append(exec_time)
        score_lst_lst.append(score_lst)
        conv_iter_lst.append(conv_iter)

    plt.clf()
    plt.plot(gamma_list, exec_time_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, exec_time_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, exec_time_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, exec_time_lst[3], label='Size - 15*15')
    plt.plot(gamma_list, exec_time_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.title('Value Iteration, Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.legend()
    plt.savefig("images/VI/VI_ExecutionTime_FL.png")
    plt.clf()

    plt.plot(gamma_list, score_lst_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, score_lst_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, score_lst_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, score_lst_lst[3], label='Size - 15*15')
    plt.plot(gamma_list, score_lst_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Value Iteration, Reward Analysis')
    plt.grid()
    plt.legend()
    plt.savefig("images/VI/VI_AverageRewards_FL.png")
    plt.clf()

    plt.plot(gamma_list, conv_iter_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, conv_iter_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, conv_iter_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, conv_iter_lst[3], label='Size - 15*15')
    plt.plot(gamma_list, conv_iter_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Value Iteration, Convergence Analysis')
    plt.grid()
    plt.legend()
    plt.savefig("images/VI/VI_Convergence_FL.png")
    plt.clf()
