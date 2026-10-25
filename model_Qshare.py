
import numpy as np
import math
import sys


import numpy as np
import random


# greedy方策
class Greedy(object):  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def select_action(self, value, env):
        act = np.asarray([0, 0])
        _act = np.where(value == np.max(value))
        # print(_act)
        select_act = np.random.randint(len(_act[0]))
        print("select_act:\t{}".format(select_act))
        for n in range(len(env.n_act)):
            act[n] = _act[n][select_act]
        print("acr:\t{}" .format(act))
        return act

    def init_params(self):
        pass

    def update_params(self):
        pass


# ε-greedy
class EpsGreedy(Greedy):
    def __init__(self, TG):
        self.TG = TG

    def select_action(self, value, N_episode):
        eps = max(1-(N_episode/self.TG), 0)
        
        if np.random.random() < eps:
            return random.randint(0, len(value)-1)
        else:
            idx = np.where(value == max(value))
            return random.choice(idx[0])

# 
import numpy as np
from copy import deepcopy

class Q_learning(object):

    def __init__(self, learning_rate, discount_rate, num_state, num_action):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_state = num_state
        self.num_action = num_action
        self.Q = np.zeros((num_state, num_action))

    def update_Q(
            self, current_state, next_state,
            current_action, reward, next_action=None, done=False):
        maxQ = 0 if done else max(self.Q[next_state])
        TD = (reward
              + self.discount_rate
              * maxQ
              - self.Q[current_state, current_action])
        self.Q[current_state, current_action] += self.learning_rate * TD

    def init_params(self):
        self.Q = np.zeros((self.num_state, self.num_action))

    def get_Q(self):
        return self.Q


class agent(object):
    def __init__(self, learning_rate, discount_rate, TG, num_state, num_action):

        self.value_func = Q_learning(learning_rate, discount_rate,
                                     num_state, num_action)
        self.policy = EpsGreedy(TG)

    def get_select_action(self, state, n_episode):
        value = self.value_func.Q[state]
        return self.policy.select_action(value, n_episode)

    def init_params(self):
        self.value_func.init_params()
        self.policy.init_params()

    def update(self, current_state, current_action, reward, next_state, done=False):
        self.value_func.update_Q(current_state, next_state, current_action, reward, done=done)

    # 各行動についてmax Q(s, a)を共有
    def share_Q(self, agents):
        s, a = agents[0].value_func.Q.shape
        value = np.zeros((len(agents), s, a))
        for n in range(len(agents)):
            value[n] = agents[n].value_func.Q
        
        self.value_func.Q = np.max(value, axis=0)
