import json
import os
import random
import sys
from time import sleep

from utils import train_test_levels

env_name = "test-map-v1"
from environment import register_baba_env

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
        _, reward, done, _ = env.step(action)
        return reward,done


if __name__ == "__main__":
    random.seed(1)
    train, test = train_test_levels()
    levels =[*train,*test]
    
    level_performance = {}
    for i,level in enumerate(levels):
        env_name = f"baba-babaisyou{i}-v0"
        register_baba_env(env_name, path=level, max_episode_steps=250)
        env = gym.make(env_name)
        env.reset()
        agent = RandomAgent()

        level_performance[level] = {"score": 0, "steps": 0, "won": False}
        try:
            score = 0
            steps = 0
            reward = 0
            done = False
            while not done:
                reward, done = agent.step(env)
                score += reward
                steps +=1
                env.render()
            level_performance[level] = {"score": score, "steps": steps, "won": reward > 0}
        except:
            continue
    with open(f"{os.path.split(__file__)[0]}/../Results/random_results.json", "w") as f:
        json.dump(level_performance,f)
