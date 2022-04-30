import random
import sys
from time import sleep
sys.path.append("../baba-is-auto/Extensions/BabaRL/baba-babaisyou-v0")
import environment
import pdb

import heapq

import rendering

import pyBaba
import numpy as np
import gym
import random
from typing import List





class AStarAgent:
    

    def __init__(self):
        self.moves = [] # the final set of most optimal moves
        self.frontier = [()] # Priority queue of what move to take next (predicted_cost, env, moves_taken)



    
    def heuristic(self, env:gym.Env) -> int:
        positions = self.get_your_positions(env)
        goals = self.get_goal_positions(env)
        avg_goal = goals.mean(axis=1)
        heuristics = []
        for pos_group in positions:
            avg_pos = pos_group.mean(axis=1)
            heuristics.append(sum(abs(avg_pos - avg_goal)))
        return heuristics



    def simulate(self, env: gym.Env) -> bool:
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
        Returns:
            Whether the environment is at a final state
        """
        done = False
        self.frontier = [(1, env, 0)]
        while not done:
            (predicted_cost, env, moves_taken) = heapq.heappop(self.frontier)
            # TODO: we need to know where we have already visited... Possibly set of tuples
            heuristics = self.heuristic(env)
            moves_taken +=1
            for h, action in zip(heuristics, env.action_space):
                predicted_cost = h + moves_taken
                possible_env = env.copy()
                _,_, done, _ = possible_env.step(action)
                # TODO: Something when done?
                entry = (predicted_cost, possible_env, moves_taken)
                heapq.heappush(entry)


    def step(self, env: gym.Env):
        action = self.moves.remove(0)
        _, _, done, _ = env.step(action)
        return done
    
    def get_your_positions(self, env: gym.Env) -> List[np.array]:
        positions = env.game.GetMap().GetPositions(env.game.GetPlayerIcon())
        positions = [np.array(p) for p in positions]
        return positions

    def get_goal_positions(self, env:gym.Env) -> List[np.array]:
        rule_manager = env.game.GetRuleManager()
        type = rule_manager.GetRules(pyBaba.WIN); 
        convert = pyBaba.ConvertTextToIcon
        win_positions =[]
        game_map = env.game.GetMap()

        map_height = game_map.GetHeight()
        map_width = game_map.GetWidth()
        for y in range(map_height): 
            for x in range(map_width): 
                if game_map.At(x,y).HasType(convert(type)):
                    win_positions.append((x,y))

        return win_positions
        


if __name__ == '__main__':
    env = gym.make('baba-babaisyou-v0')
    simulation_gym = gym.make('baba-babaisyou-v0')
    state = env.reset().reshape(1, -1, 9, 11)
    moves = 40
    done = False
    agent = AStarAgent()
    agent.simulate(simulation_gym)
    while not done:
        done = agent.step(env)
        env.render()
        sleep(0.2)
