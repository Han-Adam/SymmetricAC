import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import RollControlEnv
from mpc import AltitudeMPC
import scipy
import os
import argparse
import json


path = os.path.dirname(os.path.realpath(__file__))
path += '/'
env = RollControlEnv()
# mpc =AltitudeMPC()

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
    mpc = AltitudeMPC()
    for ep_step in range(500):
        a = mpc.solve(x_init=[env.current_pos[2], env.current_vel[2]],
                      x_ref=[0, 0])
        # a, e, vel_e = pid.computControl(env.current_pos[2], env.target)
        s_, r, done, info = env.step(a)
        print(env.current_pos)

        # P.append(e)
        # D.append(vel_e)

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

