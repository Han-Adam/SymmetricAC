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
              [4]])
Q = np.array([[1.5, 0],
              [0, 0.5]])
R = np.array([[0.5]])

k = lqr(A, B, Q, R)
print(k)
# [1.73205081 1.3660254 ]
k = k[0]

k = [1.732, 1.366]

# k = np.array([1.69, 1.348])

# A = np.array([[0, 1, 0, 0],
#               [0, 0, 5, 0],
#               [0, 0, 0, 1],
#               [0, 0, -20, -3.16]])
# B = np.array([[0],
#               [0],
#               [0],
#               [20]])
# Q = np.array([[10, 0, 0, 0],
#               [0, 100, 0, 0],
#               [0, 0, 100, 0],
#               [0, 0, 0, 100]])
# R = np.array([[30]])
#
# k = lqr(A, B, Q, R)
# print(k)
#
# k = k[0]
# [-3.16227766 -2.47192998  7.92678577  1.33890947]
path = os.path.dirname(os.path.realpath(__file__))
path += '/'
env = RollControlEnv()
# pid = PID(P=1.5, I=0, D=0.5)
# pid = PID(P=3, I=0.1, D=0.7)
pid = PID(P=1, I=0, D=0.77)
# pid = PID(P=20, I=0, D=3.19)
# pid = PID(P=40, I=0, D=19)
# 1.5 0 0.5
# 3 0 0.7
P = []
I = []
D = []

x = []
x_target = []
roll = []

y = []
y_target = []
vel = []
acc = []
pitch = []

z = []
z_target = []
yaw = []

targets = np.array([[5, 5, 0],
                    [-5, -5, 0]])
for episode in range(2):
    target = targets[episode, :]
    s = env.reset(target=target)
    for ep_step in range(500):
        a, e, vel_e = pid.computControl(env.current_pos[0], env.target[0])
        state = np.array([0-env.current_pos[0], -env.current_vel[0]])
        a = np.dot(k, state)
        # print(k, state, a)
        a = np.clip(a, -1, 1)
        s_, r, done, info = env.step(a)

        P.append(e)
        D.append(vel_e)

        x.append(env.current_pos[0])
        x_target.append(target[0])
        roll.append(env.current_ori[0])

        y.append(env.current_pos[1])
        # vel.append(env.current_vel[1])
        y_target.append(target[1])
        vel.append(env.current_vel[1])
        acc.append((env.current_vel[1]-env.last_vel[1])/0.01)
        pitch.append(env.current_ori[1])

        z.append(env.current_pos[2])
        z_target.append(target[2])
        yaw.append(env.current_ori[2])
print(k)

index = np.array(range(len(x))) * 0.01
zeros = np.zeros_like(index)
roll = np.array(roll) / np.pi*180
pitch = np.array(pitch) / np.pi*180
yaw = np.array(yaw) / np.pi*180
plt.subplot(3, 2, 1)
plt.plot(index, x, label='x')
plt.plot(index, x_target, label='x_target')
# plt.plot(index, [vel[i][0] for i in range(len(index))])
# plt.plot(index, 0.3 * np.ones_like(index))
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(index, pitch, label='pitch')
plt.plot(index, zeros)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(index, y, label='y')
plt.plot(index, y_target, label='y_target')
plt.plot(index, vel, label='vel')
plt.plot(index, acc, label='acc')
# plt.plot(index, 0.3 * np.ones_like(index))
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(index, roll, label='roll')
plt.plot(index, zeros)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(index, z, label='z')
plt.plot(index, z_target, label='z_target')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(index, yaw, label='yaw')
plt.plot(index, zeros)
plt.legend()

plt.show()
# *180/np.pi
# index = np.array(range(len(target)))*0.01
# plt.subplot(2, 1, 1)
# plt.plot(index, target, label='target')
# plt.plot(index, z, label='z')
# plt.plot(index, z_vel, label='roll_vel')
# plt.plot(index, z_acc, label='acc')
# plt.plot(index, action, label='action')
# plt.legend()
# plt.subplot(2, 1, 2)
# P = np.array(P)
# D = np.array(D)
# plt.plot(index, P*10, label='P')
# plt.plot(index, D*1.61, label='D')
# plt.plot(index, P*10+D*1.61, label='sum')
# plt.plot(index, action, label='action')
# plt.legend()
# plt.show()

x = np.array(x)
print(x.shape)
x_record = np.empty(shape=[2, 500])
x_record[0, :] = x[0:500]
x_record[1, :] = x[500:1000]
np.save(path + 'test.npy', x_record)
plt.plot(np.array(range(500)), x_record[0, :])
plt.plot(np.array(range(500)), x_record[1, :])
plt.show()

