import copy
import gym
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

import pyBaba
import pygame
import rendering
from gym import spaces

# Registration check
registered_envs = set()

def register_baba_env(name:str, path:str, enable_render=True, max_episode_steps=200, user_controls=False):
    if name not in registered_envs:
        registered_envs.add(name)
        register(id=name, entry_point='environment:BabaEnv', max_episode_steps=max_episode_steps, nondeterministic=True, kwargs={"path": path, "enable_render":enable_render, "user_controls":user_controls})


class BabaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, path:str="", enable_render=True, user_controls=False):
        super(BabaEnv, self).__init__()
        self.path = path
        self.game = pyBaba.Game(self.path)
        self.enable_render = enable_render
        
        if enable_render:
            self.renderer = rendering.Renderer(self.game)
        
        if user_controls and enable_render:
            self.setup_user_controls()

        self.action_space = spaces.Discrete(4)

        self.action_space = [
            pyBaba.Direction.UP,
            pyBaba.Direction.DOWN,
            pyBaba.Direction.LEFT,
            pyBaba.Direction.RIGHT,
        ]
        self.observation_space = spaces.MultiBinary(self.get_obs().shape)


        self.action_size = len(self.action_space)

        self.seed()
        self.reset()

    def setup_user_controls(self):
        def handle_movement(event):
            if event.unicode == "w":
                self.step(pyBaba.Direction.UP)
            elif event.unicode == "a":
                self.step(pyBaba.Direction.LEFT)
            elif event.unicode == "s":
                self.step(pyBaba.Direction.DOWN)
            elif event.unicode == "d":
                self.step(pyBaba.Direction.RIGHT)

        self.renderer.on(pygame.KEYUP,handle_movement)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        self.game.Reset()
        self.done = False

        return self.get_obs()

    def step(self, action):
        action = action if type(action) == pyBaba.Direction else self.action_space[action]
        self.game.MovePlayer(action)

        result = self.game.GetPlayState()

        if result == pyBaba.PlayState.LOST:
            self.done = True
            reward = -100
        elif result == pyBaba.PlayState.WON:
            self.done = True
            reward = 200
        else:
            reward = -0.5

        return self.get_obs(), reward, self.done, {}

    def render(self, mode='human', close=False):
        if not self.enable_render:
            raise Exception("Renderer not enabled")
        if close:
            self.renderer.quit_game()

        return self.renderer.render(self.game.GetMap(), mode)
        
    def copy(self):
        return copy.deepcopy(self.game)
    
    def set_game(self, game):
        self.game = game

    def get_obs(self):
        return np.array(
            pyBaba.Preprocess.StateToTensor(self.game),
            dtype=np.float32).reshape(-1, self.game.GetMap().GetHeight(), self.game.GetMap().GetWidth())

