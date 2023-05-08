import time
from EnvUAV.env import YawControlEnv
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()

    pos = []
    ang = []

    env.reset(base_pos=np.array([0, 0, 0]), base_ori=np.array([0, 0, 0]))

    length = 5000
    # name = 'Trajectory1'
    # index = np.array(range(length)) / length * 2
    # tx = 2 * np.sin(2 * np.pi * index) * np.cos(np.pi * index)
    # ty = 2 * np.sin(2 * np.pi * index) * np.sin(np.pi * index)
    # tz = -np.sin(2 * np.pi * index) * np.cos(np.pi * index) - np.sin(2 * np.pi * index) * np.sin(np.pi * index)
    # tpsi = np.sin(4 * np.pi * index) * np.pi / 4 * 3

    name = 'Trajectory2'
    index = np.array(range(length))/length
    tx = 2*np.cos(2*np.pi*index)
    ty = 2*np.sin(2*np.pi*index)
    tz = -np.cos(2*np.pi*index)-np.sin(2*np.pi*index)
    tpsi = np.sin(2 * np.pi * index) * np.pi / 3 * 2

    targets = np.vstack([tx, ty, tz, tpsi]).T
    for i in range(length):
        target = targets[i, :]
        env.step(target)
        pos.append(env.current_pos.tolist())
        ang.append(env.current_ori.tolist())

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    position = np.array(pos)
    px = position[:, 0]
    py = position[:, 1]
    pz = position[:, 2]
    attitude = np.array(ang)
    roll = attitude[:, 0]
    pitch = attitude[:, 1]
    yaw = attitude[:, 2]

    # np.save(file=path+'/Trajectory/quad_uav.npy', arr=position)
    # np.save(file=path+'/Trajectory/quad_target.npy', arr=targets)
    # np.save(file=path+'/Trajectory/circle_uav.npy', arr=position)
    # np.save(file=path+'/Trajectory/circle_target.npy', arr=targets)
    ax.plot(px, py, pz, label='track')
    ax.plot(tx, ty, tz, label='target')
    ax.view_init(azim=45., elev=30)
    plt.show()

    zeros = np.zeros_like(index)
    plt.subplot(3, 2, 1)
    plt.plot(index, px, label='x')
    plt.plot(index, tx, label='x_target')
    # plt.plot(index, [vel[i][0] for i in range(len(index))])
    # plt.plot(index, 0.3 * np.ones_like(index))
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label='pitch')
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, py, label='y')
    plt.plot(index, ty, label='y_target')
    # plt.plot(index, vel, label='vel')
    # plt.plot(index, acc, label='acc')
    # plt.plot(index, 0.3 * np.ones_like(index))
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label='roll')
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, py, label='z')
    plt.plot(index, ty, label='z_target')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label='yaw')
    plt.plot(index, tpsi, label='yaw_target')
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    pos = np.array(pos)
    ang = np.array(ang)
    np.save(path + '/LQR_' + name + '_pos.npy', pos)
    np.save(path + '/LQR_' + name + '_ang.npy', ang)


if __name__ == '__main__':
    main()

