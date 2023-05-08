import numpy as np


class PID:
    def __init__(self, P, I, D):
        self.max_output = 1
        self.min_output = -1
        self.control_time_step = 0.01

        self.P = P
        self.I = I
        self.D = D

        self.control_counter = 0
        self.last_e = 0
        self.integral_e = 0
        self.last_x = 0

    def reset(self):
        self.control_counter = 0
        self.last_e = 0
        self.integral_e = 0
        self.last_x = 0

    def computControl(self,
                      current_x,
                      target):
        self.control_counter += 1
        e = target - current_x
        vel_e = 0 - (current_x - self.last_x)/ self.control_time_step

        self.last_x = current_x
        self.integral_e += e * self.control_time_step
        self.integral_e = np.clip(self.integral_e, -1, 1)

        output = self.P * e + \
                 self.I * self.integral_e + \
                 self.D * vel_e
        print(e, self.integral_e, vel_e)
        return np.clip(output, -1, 1), e, vel_e