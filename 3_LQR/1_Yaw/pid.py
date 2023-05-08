import numpy as np
import pybullet as p


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

    def computControl(self,
                      ang,
                      target,
                      ang_vel):
        self.control_counter += 1
        R = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ang)), [3, 3])
        R_d = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(target)), [3, 3])
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d))/2
        e = e_R[0,1] #x:[1,2], y[2, 0], z[0,1]

        # e = target - current_x
        vel_e = ang_vel[2]
        output = self.P * e - self.D * vel_e
        return np.clip(output, -1, 1)