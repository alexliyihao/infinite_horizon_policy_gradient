import numpy as np

class environment():
    def __init__(self):
        self.current = None
        self.state_space = []
        self.action_space = []

    def env_return(self, action):
        pass

    def __call__(self, action):
        current_state, reward = self.env_return(action)
        return current_state, reward

    def get_current(self):
        return self.current

    def set_current(self, state):
        self.current = state

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

class riverswim(environment):
    def __init__(self):
        self.state_space = np.array([0,1,2,3,4,5])
        self.action_space = np.array([0,1])
        self.current = np.random.choice([0,1],p = [0.5,0.5])

    def reset(self):
        self.current = np.random.choice([0,1],p = [0.5,0.5])
        return self.current

    def env_return(self, action):
        assert action in self.action_space
        assert self.current in self.state_space

        if self.current == 0:
            if action == 0:
                self.current = 0
                reward = 0.0005
            if action == 1:
                self.current += np.random.choice([0,1], p=[0.7, 0.3])
                reward = 0
        elif self.current in [1,2,3,4]:
            if action == 0:
                self.current -= 1
                reward = 0
            if action == 1:
                self.current += np.random.choice([-1,0,1], p=[0.1, 0.3, 0.6])#[0.1, 0.6, 0.3])
                reward = 0
        elif self.current == 5:
            if action == 0:
                self.current -= 1
                reward = 0
            if action == 1:
                self.current += np.random.choice([-1,0], p = [0.7, 0.3])
                if self.current == 5:
                    reward = 1
                else:
                    reward = 0
        return self.current, reward

    def get_reward(self, state, action, next_state):
        if state == 0:
            if action == 0:
                reward = 0.0005
            if action == 1:
                reward = 0
        elif state in [1,2,3,4]:
              reward = 0
        elif state == 5:
            if action == 1 and next_state == 5:
                reward = 1
            else:
                reward = 0
        return reward
