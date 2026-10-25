import numpy as np

class GlidWorld(object):
    # 簡単なステージを作る
    def __init__(self, row, col, start, goal=None):
        self.row = row
        self.col = col
        self.start_state = start
        self.goal_state = goal
        self.num_state = row * col
        self.num_action = 4
        self.reward = 0
        self.done = False

    # 座標に変換
    def coord_to_state(self, row, col):
        return ((row * self.col) + col)

    # 座標からx軸を算出
    def state_to_row(self, state):
        return ((int)(state / self.col))

    # 座標からy軸を算出
    def state_to_col(self, state):
        return (state % self.col)

    # 次の座標を算出
    def evaluate_next_state(self, state, action):
        UPPER = 0
        LOWER = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4

        row = self.state_to_row(state)
        col = self.state_to_col(state)

        if action == UPPER:
            if (row) > 0:
                row -= 1
        elif action == LOWER:
            if (row) < (self.row-1):
                row += 1
        elif action == RIGHT:
            if (col) < (self.col-1):
                col += 1
        elif action == LEFT:
            if (col) > 0:
                col -= 1
        elif action == STOP:
            pass

        self.next_state = self.coord_to_state(row, col)

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal_state:
            self.reward = 1
        else:
            self.reward = 0

    def evaluate_done(self):
        if self.reward > 0:
            self.done = True
        else:
            self.done = False

    # 行動
    def step(self, state, action):
        self.evaluate_next_state(state, action)
        self.evaluate_reward(self.next_state)
        self.evaluate_done()

        return self.next_state, self.reward, self.done
    

class SuboptimaWorld(GlidWorld):
    DEFAULT_REWARD_LAYOUT = [8, 1, 3, 5, 6, 4, 2, 7]

    def __init__(self, row, col, start, goal=None, reward_values=None):
        super().__init__(row, col, start, goal=None)

        self.world = np.zeros([row, col])
        self.goal_coordinates = [
            (0, 2),
            (0, col - 3),
            (2, 0),
            (2, col - 1),
            (row - 3, 0),
            (row - 3, col - 1),
            (row - 1, 2),
            (row - 1, col - 3),
        ]
        if reward_values is None:
            reward_values = self.DEFAULT_REWARD_LAYOUT
        if len(reward_values) != len(self.goal_coordinates):
            raise ValueError("reward_values must have exactly 8 values.")

        for (goal_row, goal_col), reward in zip(self.goal_coordinates, reward_values):
            self.world[goal_row, goal_col] = reward

    def evaluate_reward(self, state):
        row = self.state_to_row(state)
        col = self.state_to_col(state)
        self.reward = self.world[row, col]
        return self.reward
