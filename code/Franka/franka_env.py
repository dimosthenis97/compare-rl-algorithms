import numpy as np
import RobotDART as rd
import gym
from gym import spaces
import warnings

warnings.filterwarnings("ignore")


class Frankaenv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dt=0.01,
        target_positions=np.array(
            [
                np.pi / 3.0,
                np.pi / 2.0,
                np.pi / 1.5,
                np.pi / 3.0,
                np.pi / 2.0,
                np.pi,
                np.pi / 3.0,
                0.03,
                0.03,
            ]
        ),
    ):
        self._dt = dt

        # robot,robot position
        packages = [("franka_description", "franka/franka_description")]
        self._robot = rd.Robot("franka/franka.urdf", packages)
        self._robot.set_color_mode("material")

        self._robot.fix_to_world()
        self._robot.set_position_enforced(True)
        self._robot.set_actuator_types("servo")

        positions = self._robot.positions()
        positions[5] = np.pi / 2.0
        positions[6] = np.pi / 4.0
        positions[7] = 0.04
        positions[8] = 0.04
        self._robot.set_positions(positions)

        pos = (
            self._robot.positions()
            + np.random.rand(self._robot.num_dofs()) * np.pi / 1.5
            - np.pi / 3.0
        )
        pos[7] = pos[8] = 0.03
        self._robot.set_positions(pos)

        self._target_positions = target_positions
        self._current = np.zeros(target_positions.shape[0])
        self._current[3] = np.pi / 1.5



        # simulation
        self._simu = rd.RobotDARTSimu(dt)
        self._simu.set_collision_detector("fcl")

        ########    graphics    ##############

        gconfig = rd.gui.GraphicsConfiguration(1024, 768)
        graphics = rd.gui.Graphics(gconfig)
        self._simu.set_graphics(graphics)
        graphics.look_at([3.0, 1.0, 2.0], [0.0, 0.0, 0.0])

        ######################
        self._simu.add_checkerboard_floor()
        self._simu.add_robot(self._robot)
        self._countstep = 0
        self._count = 0
        self._t = 0
        self._reward = -100

        self._done = False

        self.action_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(9,), dtype=np.float32
        )

    def step(self, action):

        self._countstep = self._countstep + 1

        self._robot.set_commands(action, self._robot.dof_names(True, True, True)[:-1])
        self._simu.step_world()


        self._current = self._robot.positions()
        vec = self._target_positions - self._current

        reward_dis = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        self._reward = reward_dis + reward_ctrl

        # evry 10000 reset env
        if self._countstep == 800:
            self._countstep = 0
            self._done = True

        info = {}
        return self._current, self._reward, self._done, info

    def reset(self):
        self._robot.reset()
        positions = self._robot.positions()
        positions[5] = 0
        positions[6] = 0
        self._robot.set_positions(positions)

        pos = (
            self._robot.positions()
            + np.random.rand(self._robot.num_dofs()) * np.pi / 1.5
            - np.pi / 3.0
        )

        self._robot.set_positions(pos)

        self._current = self._robot.positions()
        self._done = False
        return self._current


#     print(current, cmd)
