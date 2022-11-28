import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from gym.envs.toy_text import frozen_lake
import time

# https://gist.github.com/jojonki/6291f8c3b19799bc2f6d5279232553d7
# Q learning params
ALPHA = 0.1  # learning rate
#GAMMA = 0.999  # reward discount
LEARNING_COUNT = 20000
TEST_COUNT = 1000

TURN_LIMIT = 20000


class Agent:
    def __init__(self, env, gamma, map_size):
        self.env = env
        self.gamma = gamma
        self.episode_reward = 0.0
        self.epsilon = 1.0
        self.q_val = np.zeros(map_size * map_size * 4).reshape(map_size * map_size, 4).astype(np.float32)
        # exploartion decreasing decay for exponential decreasing
        self.epsilon_decay = 0.9999
        # minimum of exploration proba
        self.epsilon_min = 0.01

    def learn(self):
        # one episode learning
        state = self.env.reset()
        total_reward = 0.0
        # self.env.render()

        for t in range(TURN_LIMIT):
            pn = np.random.random()
            if pn < self.epsilon:
                act = self.env.action_space.sample()  # random
            else:
                act = self.q_val[state].argmax()
            next_state, reward, done, info = self.env.step(act)
            total_reward += reward
            q_next_max = np.max(self.q_val[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act] + ALPHA * (
                        reward + self.gamma * q_next_max - self.q_val[state, act])
            # self.env.render()
            if done or t == TURN_LIMIT - 1:
                return total_reward
            else:
                state = next_state
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def test(self):
        state = self.env.reset()
        total_reward = 0.0
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state])
            next_state, reward, done, info = self.env.step(act)
            total_reward += reward
            if done or t == TURN_LIMIT - 1:
                return total_reward
            else:
                state = next_state
        return 0.0  # over limit

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
    plt.savefig('images/Q/'+title + str('.png'))
    #plt.show()
    plt.clf()


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
    learn_score_lst = []
    test_score_lst = []
    conv_iter_lst = []

    for size in [4, 8, 10, 15]:
        #size = 20
        exec_time = []
        learn_score = []
        test_score = []
        conv_iter = []
        policy = []
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
            agent = Agent(env, gamma, size)
            # Learning
            reward_total = 0.0
            for i in range(LEARNING_COUNT):
                reward = agent.learn()
                reward_total += reward
                if i % 10000 == 0:
                    print('Learn', option, str(i))

            end = time.time()
            learn_score.append(reward_total / LEARNING_COUNT)
            exec_time.append(end - start_time)

            # Test
            reward_total = 0.0
            for i in range(TEST_COUNT):
                reward = agent.test()
                reward_total += reward

            test_score.append(reward_total / TEST_COUNT)

            policy_curr = [np.argmax(agent.q_val[state]) for state in range(size * size)]
            policy_curr = np.array(policy_curr)
            policy.append(policy_curr)

            show_policy_map(
                'Frozen Lake  ' + option + ' Q-Learning - Gamma ' + str(gamma),
                policy_curr.reshape(size, size), desc, colors(), directions())

        exec_time_lst.append(exec_time)
        learn_score_lst.append(learn_score)
        test_score_lst.append(test_score)
    plt.clf()
    plt.plot(gamma_list, exec_time_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, exec_time_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, exec_time_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, exec_time_lst[3], label='Size - 15*15')
    #plt.plot(gamma_list, exec_time_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.title('Gamma vs Execution Time - Q-Learning')
    plt.ylabel('Execution Time (s)')
    #plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/Q/Q_ExecutionTime_FL.png")
    #plt.show()
    plt.clf()

    plt.plot(gamma_list, learn_score_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, learn_score_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, learn_score_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, learn_score_lst[3], label='Size - 15*15')
    #plt.plot(gamma_list, learn_score_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Gamma vs Rewards(Learn) - Q-Learning')
    #plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/Q/Q_LearnAverageRewards_FL.png")
    #plt.show()
    plt.clf()

    plt.plot(gamma_list, test_score_lst[0], label='Size - 4*4')
    plt.plot(gamma_list, test_score_lst[1], label='Size - 8*8')
    plt.plot(gamma_list, test_score_lst[2], label='Size - 10*10')
    plt.plot(gamma_list, test_score_lst[3], label='Size - 15*15')
    #plt.plot(gamma_list, test_score_lst[4], label='Size - 20*20')
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Gamma vs Rewards(Test) - Q-Learning')
    #plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/Q/Q_TestAverageRewards_FL.png")
    #plt.show()
    plt.clf()