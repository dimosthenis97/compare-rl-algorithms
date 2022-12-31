from cmath import cos
from sre_parse import State
import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
import gym
from gym import spaces
from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import break_cycles
from utils import angle_wrap_multi


class Pendubot(gym.Env):
    def __init__(
        self,
        dt=0.01,
        target_positions=np.array([np.pi, 0.0]),
    ):
        self._dt = dt

        # robot,robot position
        self._robot = rd.Robot("pendubot.urdf")
        self._robot.set_actuator_types("torque")
        positions = self._robot.positions()
        positions[0] = 0
        positions[1] = 0
        self._robot.set_positions([0, 0])
        self._min = 0
        self._current = np.zeros(2)
        self._target_positions = target_positions

        # simulation
        self._simu = rd.RobotDARTSimu(dt)
        self._simu.add_robot(self._robot)

        # # ################graphics###############
        gconfig = rd.gui.GraphicsConfiguration(1024, 768)
        graphics = rd.gui.Graphics(gconfig)
        self._simu.set_graphics(graphics)
        graphics.look_at([0.0, 3.0, 2.0], [0.0, 0.0, 0.0])

        #########################################

        self._robot.fix_to_world()
        self._done = False
        self._count = 0
        self._reward = -100
        # self.state = [0.0, 0.0, 0.0, 0.0]
        self._th1vel, self._th2vel = 0.0, 0.0

        self.action_space = spaces.Box(
            low=-1000, high=1000, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(6,), dtype=np.float32
        )

    def step(self, action):
        self._count = self._count + 1

        self._robot.set_commands(action, self._robot.dof_names(True, True, True)[:-1])
        self._simu.step_world()

        th1, th2 = self._robot.positions()


        self._th1vel, self._th2vel = self._robot.velocities()

        self.state = np.array(
            [
                np.cos(th1),
                np.sin(th1),
                np.cos(th2),
                np.sin(th2),
                self._th1vel,
                self._th2vel,
            ]
        )
        # to (- np.pi / 2) einai gia to swsto orthokanoniko susthma
        end_effector_x = np.cos(th1 - np.pi / 2) + np.cos(
            (th1 - np.pi / 2) + (th2 - np.pi / 2)
        )
        end_effector_y = np.sin(th1 - np.pi / 2) + np.sin(
            (th1 - np.pi / 2) + (th2 - np.pi / 2)
        )
        target_position_x = np.cos(np.pi - np.pi / 2) + np.cos(
            (np.pi - np.pi / 2) + (0 - np.pi / 2)
        )
        target_position_y = np.sin(np.pi - np.pi / 2) + np.sin(
            (np.pi - np.pi / 2) + (0 - np.pi / 2)
        )
        self._reward = -np.sqrt(
            (target_position_x - end_effector_x) ** 2
            + (target_position_y - end_effector_y) ** 2
        )

        if self._count == 200:
            self._count = 0
            self._done = True

        info = {}
        return (
            np.array(self.state),
            self._reward,
            self._done,
            info,
        )

    def reset(self):
        self._robot.reset()
        positions = self._robot.positions()
        positions[0] = 0
        positions[1] = 0
        self._robot.set_positions([0, 0])

        self.state = [1, 0.0, 1, 0.0, 0.0, 0.0]

        self._done = False
        return self.state
        
        
        
        
      
