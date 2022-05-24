import random
import sys
from time import sleep

env_name = "test-map-v1"
sys.path.append("..")
from environment import register_baba_env
sys.path.append("../baba-is-auto/Extensions/BabaRL/baba-babaisyou-v0")

import rendering

import pyBaba
import numpy as np
import gym
import random


class RandomAgent:
    def step(self, env: gym.Env) -> bool:
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
        Returns:
            Whether the environment is at a final state
        """
        action = random.choice(env.action_space)
        _, _, done, _ = env.step(action)
        return done


if __name__ == "__main__":
    env_template = register_baba_env(env_name, path=f"../levels/out/0.txt")
    env = gym.make(env_name)
    env.reset()
    moves = 40
    done = False
    agent = RandomAgent()
    env.render()
    for i in range(moves):
        if done:
            break
        agent.step(env)
        # env.render()
        sleep(0.2)
