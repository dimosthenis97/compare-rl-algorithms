from cgitb import reset
import numpy as np
import RobotDART as rd
import gym
from gym import spaces
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt
from numpy.testing._private.utils import break_cycles
import os


class Myenv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dt=0.015,
        target_positions=np.array([-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2]),
        start_state=np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2]),
        reward=-100,
    ):

        self._dt = dt
        self._done = False

        self._simu = rd.RobotDARTSimu(dt)
        self._robot = rd.Robot("arm.urdf")

        self._robot.fix_to_world()
        self._robot.set_positions([-0.2, -0.2, -0.2, -0.2])
        self._robot.set_actuator_types("servo")
        self._current = self._robot.positions()
        self._reward = reward
        self._start_state = start_state
        self._target_positions = target_positions
        self._countstep = 0
        self._count = 0
        ################     graphics
        gconfig = rd.gui.GraphicsConfiguration(1024, 768)
        graphics = rd.gui.Graphics(gconfig)
        self._simu.set_graphics(graphics)

        self._simu.add_checkerboard_floor()

        ################


        self._simu.add_robot(self._robot)

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

    def step(self, action):
        self._countstep = self._countstep + 1
        # action
        self._robot.set_commands(action)
        self._simu.step_world()


        self._current = self._robot.positions()

        th1, th2, th3, th4 = self._robot.positions()
        end_effector_x = (
            0.123 * np.cos(th1)
            + 0.317 * np.cos(th1 + th2)
            + 0.202 * np.cos(th1 + th2 + th3)
            + 0.1605 * np.cos(th1 + th2 + th3 + th4)
        )

        end_effector_y = (
            0.123 * np.sin(th1)
            + 0.317 * np.sin(th1 + th2)
            + 0.202 * np.sin(th1 + th2 + th3)
            + 0.1605 * np.sin(th1 + th2 + th3 + th4)
        )
        target_x = (
            0.123 * np.cos(np.pi / 2)
            + 0.317 * np.cos(np.pi / 2 + np.pi / 3)
            + 0.202 * np.cos(np.pi / 2 + np.pi / 3 + np.pi / 6)
            + 0.1605 * np.cos(np.pi / 2 + np.pi / 3 + np.pi / 6 + np.pi / 2)
        )
        target_y = (
            0.123 * np.sin(np.pi / 2)
            + 0.317 * np.sin(np.pi / 2 + np.pi / 3)
            + 0.202 * np.sin(np.pi / 2 + np.pi / 3 + np.pi / 6)
            + 0.1605 * np.sin(np.pi / 2 + np.pi / 3 + np.pi / 6 + np.pi / 2)
        )

        # reward

        self._reward = -np.sqrt(
            (target_x - end_effector_x) ** 2 + (target_y - end_effector_y) ** 2
        )

        # evry 10000 reset env
        if self._countstep == 800:
            self._countstep = 0
            self._done = True

        # return
        info = {}
        return self._current, self._reward, self._done, info

    def reset(self):
        self._robot.reset()

        self._done = False

        self._robot.set_positions([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2])
        self._current = self._robot.positions()

        return self._current
