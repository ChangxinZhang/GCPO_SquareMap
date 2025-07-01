import numpy as np


class SquareMapEnv:
    def __init__(self, scale_mode):
        self.width = 10
        self.height = 10
        self.ini_state = (1.5/10, 1.5/10, 0, 0)

        self.danger_zones = [
            (2.5, 3.5, 6.5, 7.5),
            (7.5, 0.5, 9.5, 2.5),
            (0, 0, 0.5, 10),
            (0.5, 9.5, 10, 10),
            (9.5, 0, 10, 9.5),
            (0.5, 0, 9.5, 0.5)
        ]

        self.target_zones = [(8, 8, 9, 9)]

        if scale_mode == 1:
            self.task_reward = 1
            self.step_reward = -0.01
            self.danger_cost = 1
        elif scale_mode == 2:
            self.task_reward = 10
            self.step_reward = -0.1
            self.danger_cost = 1
        elif scale_mode == 3:
            self.task_reward = 100
            self.step_reward = -1
            self.danger_cost = 0.1

        self.max_speed = 0.5
        self.max_acceleration = 0.2

    def compute_reward(self, x, y, done):

        for zone in self.target_zones:
            x1, y1, x2, y2 = zone
            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                reward = self.task_reward
                done = True
            elif x < 0 or x >= self.width or y < 0 or y >= self.height:
                done = True
                reward = 0
            else:
                reward = 0
        return reward, done

    def step(self, action):
        done = False
        cost = 0
        action = self.max_acceleration * action
        action_x = action[0]
        action_y = action[1]

        x, y, x_velocity, y_velocity = self.state
        x = x * self.width
        y = y * self.height
        x_velocity = x_velocity * self.max_speed
        y_velocity = y_velocity * self.max_speed

        speed = np.sqrt(x_velocity**2 + y_velocity**2)
        if speed > self.max_speed:
            x_velocity *= self.max_speed / speed
            y_velocity *= self.max_speed / speed

        x += x_velocity
        y += y_velocity

        acceleration = np.sqrt(action_x**2 + action_y**2)
        if acceleration > self.max_acceleration:
            action_x *= self.max_acceleration / acceleration
            action_y *= self.max_acceleration / acceleration

        x_velocity += action_x
        y_velocity += action_y

        reward, done = self.compute_reward(x, y, done)
        in_danger = False
        for zone in self.danger_zones:
            x1, y1, x2, y2 = zone
            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                in_danger = True
                break
        if in_danger:
            cost = self.danger_cost

        x = x / self.width
        y = y / self.height
        x_velocity = x_velocity / self.max_speed
        y_velocity = y_velocity / self.max_speed
        self.state = (x, y, x_velocity, y_velocity)

        return self.state, reward, cost, done, {}

    def reset(self):
        self.state = self.ini_state
        return self.state
