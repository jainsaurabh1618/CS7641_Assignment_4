import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from gym.envs.toy_text import frozen_lake
import time


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
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


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    #eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    max_iterations = 2000
    #gamma = 1.0
    k = 0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            k = i+1
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy, k

# https://www.kaggle.com/code/arjunayyangar/assignment4-1-frozen-lake/notebook
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
    plt.savefig('images/PI/'+title + str('.png'))
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
    #env_name  = 'FrozenLake8x8-v0'
    env_name = 'FrozenLake-v0'
    np.random.seed(42)

    exec_time_lst = []
    score_lst_lst = []
    conv_iter_lst = []

    for size in [4, 8, 10, 15, 20]:
        #size = 20
        exec_time = []
        score_lst = []
        conv_iter = []
        gamma_list = [0.2, 0.4, 0.8, 0.9, 0.99]

        custom_map = frozen_lake.generate_random_map(size=size, p=0.9)
        option = str(size) + "*" + str(size)

        for i, gamma in enumerate(gamma_list):

            #env = gym.make("FrozenLake-v1", desc=custom_map)
            env = gym.make(env_name, desc=custom_map)
            env.reset()
            #env.render()
            env = env.unwrapped
            desc = env.unwrapped.desc
            start_time = time.time()
            optimal_policy, k = policy_iteration(env, gamma = gamma)
            scores = evaluate_policy(env, optimal_policy, gamma = gamma)
            exec_time.append(round(time.time() - start_time, 2))
            score_lst.append(scores)
            conv_iter.append(k)
            print(option, ' : ', gamma, ' : ', k, ' : ', round(scores, 3))
            #print('Average scores = ', scores)
            show_policy_map(
                'Frozen Lake  ' + option + ' PI Policy Map - Iteration ' + str(k) + ' Gamma -  ' + str(gamma),
                optimal_policy.reshape(size, size), desc, colors(), directions())
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
    plt.title('Policy Iteration, Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.legend()
    plt.savefig("images/PI/PI_ExecutionTime_FL.png")
    plt.clf()

    plt.plot(gamma_list, score_lst_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, score_lst_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, score_lst_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, score_lst_lst[3], label='Size - 15*15')
    plt.plot(gamma_list, score_lst_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Policy Iteration, Reward Analysis')
    plt.grid()
    plt.legend()
    plt.savefig("images/PI/PI_AverageRewards_FL.png")
    plt.clf()

    plt.plot(gamma_list, conv_iter_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, conv_iter_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, conv_iter_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, conv_iter_lst[3], label='Size - 15*15')
    plt.plot(gamma_list, conv_iter_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Policy Iteration, Convergence Analysis')
    plt.grid()
    plt.legend()
    plt.savefig("images/PI/PI_Convergence_FL.png")
    plt.clf()