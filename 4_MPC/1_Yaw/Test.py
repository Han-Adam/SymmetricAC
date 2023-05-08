import numpy as np
import matplotlib.pyplot as plt
from EnvUAV.env import RollControlEnv
from pid import PID
from mpc import AttitudeMPC
import scipy
import os
import argparse
import json


env = RollControlEnv()
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

action = []

target = np.array([0, 0, np.pi/3])
# target = np.array([0, 0, np.pi/11*10])
s = env.reset()

for i in range(1):
    mpc = AttitudeMPC()
    for ep_step in range(200):
        a = mpc.solve(ang=env.current_ori,
                      target=target,
                      vel=env.current_ang_vel)
        # a = pid.computControl(env.current_ori, target, env.current_ang_vel)
        s_, r, done, info = env.step(a)

        action.append(a)

        x.append(env.current_pos[0])
        x_target.append(0)
        roll.append(env.current_ori[0])

        y.append(env.current_pos[1])
        # vel.append(env.current_vel[1])
        y_target.append(0)
        vel.append(env.current_vel[1])
        acc.append((env.current_vel[1] - env.last_vel[1]) / 0.01)
        pitch.append(env.current_ori[1])

        z.append(env.current_pos[2])
        z_target.append(0)
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
# plt.plot(index, vel, label='vel')
# plt.plot(index, acc, label='acc')
# plt.plot(index, 0.3 * np.ones_like(index))
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(index, roll, label='roll')
plt.plot(index, zeros)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(index, z, label='z')
plt.plot(index, z_target, label='z_target')
plt.plot(index, action)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(index, yaw, label='yaw')
plt.plot(index, np.ones_like(yaw)*target[2]/np.pi*180)
plt.plot(index, zeros)
plt.legend()

print(yaw - np.ones_like(yaw)*target[2]/np.pi*180)

plt.show()

