import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import RollControlEnv
from pid import PID
from mpc import PositionMPC
import scipy
import os
import argparse
import json


path = os.path.dirname(os.path.realpath(__file__))
path += '/'
env = RollControlEnv()
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
    mpc = PositionMPC()
    target = targets[episode, :]
    s = env.reset(target=target)
    for ep_step in range(500):
        print(ep_step)
        a = mpc.solve(x_init=[env.current_pos[0], env.current_vel[0]],
                      x_ref=[env.target[0], 0])
        # print(k, state, a)
        a = np.clip(a, -1, 1)
        s_, r, done, info = env.step(a)


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
print(x_record)
plt.show()

