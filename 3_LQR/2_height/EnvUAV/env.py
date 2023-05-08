import os
from .uav import UAV
from .surrounding import Surrounding
import numpy as np
import pybullet as p


class RollControlEnv:
    def __init__(self,
                 model='cf2x',
                 render=False,
                 random=True,
                 time_step=0.01):
        '''
        :param model: The model/type of the uav.
        :param render: Whether to render the simulation process
        :param random: Whether to use random initialization setting
        :param time_step: time_steps
        '''
        self.render = render
        self.model = model
        self.random = random
        self.time_step = time_step
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.client = None
        self.time = None
        self.surr = None
        self.current_pos = self.last_pos = None
        self.current_ori = self.last_ori = None
        self.current_matrix = self.last_matrix = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target = None
        self.uav = None

    def close(self):
        p.disconnect(self.client)

    def reset(self, target=None):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p. GUI if self.render else p.DIRECT)
        self.time = 0.
        # 构建场景
        self.surr = Surrounding(client=self.client,
                                time_step=self.time_step)
        # 初始化时便最好用float
        base_pos = np.array([0., 0., target])
        base_ori = np.array([0., 0., 0.])
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ori = self.last_ori = np.array(base_ori)
        self.current_matrix = self.last_matrix = np.array([[1., 0., 0.],
                                                           [0., 1., 0.],
                                                           [0., 0., 1.]])
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.target = 0
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))
        return self._get_s()

    def step(self, a):
        self.last_pos = self.current_pos
        self.last_ori = self.current_ori
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel

        self.uav.apply_action(a, self.time)
        p.stepSimulation()
        self.time += self.time_step

        current_pos, current_ori = p.getBasePositionAndOrientation(self.uav.id)
        current_matrix = np.reshape(p.getMatrixFromQuaternion(current_ori), [3, 3])
        current_ori = p.getEulerFromQuaternion(current_ori)
        current_vel, current_ang_vel = p.getBaseVelocity(self.uav.id)
        # 在环境当中，我们均以np.array的形式来存储。
        self.current_pos = np.array(current_pos)
        self.current_ori = np.array(current_ori)
        self.current_matrix = current_matrix
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.matmul(current_ang_vel, current_matrix)

        # self._check_collision()
        s_ = self._get_s()
        r = self._get_r()
        done = False
        infor = None
        return s_, r, done, infor

    def _get_s(self):
        z = self.current_pos[2]
        z_v = self.current_vel[2]
        z_acc = (self.current_vel[2] - self.last_vel[2]) / self.time_step
        target = self.target

        s = [target-z, z_v, z_acc]
        return s

    def _get_r(self):
        last_z = self.last_pos[2]
        current_z = self.current_pos[2]
        target = self.target
        r = (abs(target - last_z) - abs(target - current_z))
        return r

