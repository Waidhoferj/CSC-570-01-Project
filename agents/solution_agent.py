""" An agent that plays through a level using the provided solution """

from atexit import register
import sys
from time import sleep

env_name = "test-map-v1"
from environment import register_baba_env

import rendering

import pyBaba
import numpy as np
import gym
import random

level = 10 # The chosen level
ext = ".txt"

baba_dir_dict = {
    "U": pyBaba.Direction.UP,
    "R": pyBaba.Direction.RIGHT,
    "L": pyBaba.Direction.LEFT,
    "D": pyBaba.Direction.DOWN
}

class SolutionAgent:
    def step(self, env: gym.Env, action) -> bool:
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
            action: The move that the agent needs to make for the solution.
        Returns:
            Whether the environment is at a final state
        """
        
        try:
            _, _, done, _ = env.step(baba_dir_dict[action])
        except Exception as e:
            print(f"Error {e} -- Could not complete agent task")
            sys.exit(-1)
        return done

if __name__ == "__main__":
    print(f"Running 'solution_agent.py'\n--------------------")
    
    solution_file = f"levels/out/{level}_sol{ext}"
    level_file = f"levels/out/{level}{ext}"

    solution = ""
    with open(solution_file, 'r') as sol_file:
        solution = sol_file.read()
    
    if solution == "":
        print("Error: Solution not present.")
        sys.exit(-1)
    
    print(f"Using solution: {solution}") # DEBUG: print out solution

    env_template = register_baba_env(env_name, path=level_file)
    env = gym.make(env_name)
    env.reset()
    done = False
    agent = SolutionAgent()

    for move in solution:
        if done:
            break
        agent.step(env, move)
        env.render()
        sleep(0.2)
        