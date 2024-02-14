import gym
import gym.spaces
import panda_gym
from panda_gym.envs.panda_tasks.panda_reach import PandaReachEnv

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from panda_gym.pybullet import PyBullet
import gym.utils.seeding


from typing import Any, Dict, Optional, Tuple, Union



#prob need to ultimately rewrite this but...? 

class PandaReachDiffEnv(RobotTaskEnv):

    def __init__(self, render_size=120, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        self.sim = PyBullet(render=render)
        self.robot = Panda(self.sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = Reach(self.sim, reward_type=reward_type, get_ee_position=self.robot.get_ee_position)
        obs = self.reset()  # required for init; seed can be changed later
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        super().__init__(self.robot, self.task)
        self.observation_space = gym.spaces.Dict({
            'observation':gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
            'desired_goal':gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
            'achieved_goal':gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            'image':gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(3,render_size,render_size),
                    dtype=np.float32)
        }
        )
        self.render_cache=None

        

    def _get_obs(self) -> Dict[str, np.ndarray]:
        img = self.sim.render(mode = 'rgb_array',
                            width = 120, 
                            height= 120,
                            target_position = None,
                            distance = 0.75,
                            yaw = 45, #45
                            pitch= -45,
                            roll = 0)
        #import pdb; pdb.set_trace()
        img = np.delete(img,3,axis=2)
        #img = np.moveaxis(img.astype(np.float32) / 255, -1, 0)

        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        return {
            "observation": observation,
            "desired_goal": self.task.get_goal(),
            "achieved_goal": achieved_goal,
            "image": img
        }
    

