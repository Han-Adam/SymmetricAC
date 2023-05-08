import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import RollControlEnv
from pid import PID
import scipy
import os
import argparse
import json


def lqr(A, B, Q, R):
    '''
    dx/dt = Ax +Bu
    cost = integral x.T* Q* x + u.T* R* u
    '''
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    return np.asarray(K)


A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [5]])
Q = np.array([[10, 0],
              [0, 1.5]])
R = np.array([[1.5]])

k = lqr(A, B, Q, R)
print(k)
# exit()
# 2.58 1.42




path = os.path.dirname(os.path.realpath(__file__))
path += '/'
env = RollControlEnv()
pid = PID(P=2.58, I=0, D=1.42)

# pid = PID(P=1.5, I=0, D=0.5)
# pid = PID(P=3, I=0.1, D=0.7)
# pid = PID(P=20, I=0, D=10.5)
# pid = PID(P=20, I=0, D=3.19)
# pid = PID(P=40, I=0, D=19)
# 1.5 0 0.5
# 3 0 0.7
P = []
I = []
D = []
z = []
z_vel = []
z_acc = []
action = []
target = []
targets = np.array([5, -5])
for i in range(2):
    s = env.reset(target=targets[i])
    pid.reset()
    for ep_step in range(500):
        a, e, vel_e = pid.computControl(env.current_pos[2], env.target)
        s_, r, done, info = env.step(a)

        P.append(e)
        D.append(vel_e)

        z.append(env.current_pos[2])
        z_vel.append(env.current_vel[2])
        z_acc.append((env.current_vel[2]-env.last_vel[2])/0.01)
        # print(a)
        # print(env.current_matrix)
        target.append(env.target)
        action.append(a)
# *180/np.pi
index = np.array(range(len(target)))*0.01
# plt.subplot(2, 1, 1)
plt.plot(index, target, label='target')
plt.plot(index, z, label='z')
plt.plot(index, z_vel, label='roll_vel')
plt.plot(index, z_acc, label='acc')
plt.plot(index, action, label='action')
plt.legend()
# plt.subplot(2, 1, 2)
# P = np.array(P)
# D = np.array(D)
# plt.plot(index, P*10, label='P')
# plt.plot(index, D*1.61, label='D')
# plt.plot(index, P*10+D*1.61, label='sum')
# plt.plot(index, action, label='action')
# plt.legend()
plt.show()


x = np.array(z)
print(x.shape)
x_record = np.empty(shape=[2, 500])
x_record[0, :] = x[0:500]
x_record[1, :] = x[500:1000]
np.save(path + 'test.npy', x_record)
plt.plot(np.array(range(500)), x_record[0, :])
plt.plot(np.array(range(500)), x_record[1, :])
plt.show()

