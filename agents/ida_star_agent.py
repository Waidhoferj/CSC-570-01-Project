import random
import os
from time import sleep

env_name = "baba-volcano-v0"
env_path = os.path.join("baba-is-auto", "Resources", "Maps", "volcano.txt")
from environment import register_baba_env
import pdb

import heapq

import rendering
import pdb
import pyBaba
import numpy as np
import gym
import random
from typing import List
import time


class IDAStarAgent:
    def __init__(self):
        self.optimal_moves = []  # the final set of most optimal moves

    # combine different heuristics such as being me a goal or reaching a goal
    def heuristic(self, env: gym.Env) -> int:
        positions = self.get_your_positions(env)
        goal_positions = self.get_goal_positions(env)

        # Manhattan distance
        min_dist = min(
            np.sum(np.abs(goal_positions - pos), axis=1).min() for pos in positions
        )
        # pdb.set_trace()
        return min_dist

    def get_env_game_state(self, env):
        game_play_state = env.game.GetPlayState()
        game_objects = env.game.GetMap().GetObjects()
        game_rules = env.game.GetRuleManager().GetPropertyRules()
        game_player_icon = env.game.GetPlayerIcon()

        return (game_play_state, game_objects, game_rules, game_player_icon)

    def simulate(self, env: gym.Env) -> bool:
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
        Returns:
            Whether the environment is at a final state
        """

        threshold = self.heuristic(env)

        moves_taken = 0
        visited = []
        optimal_moves = []

        while True:
            temp = self.search(env, moves_taken, threshold, optimal_moves, visited)

            # when the goal is found
            if temp == -1:
                self.optimal_moves = optimal_moves
                print(self.optimal_moves)
                return

            # can't find the optimal moves
            if temp == np.inf:
                return

            threshold = temp

    def search(self, env, g_score, threshold, optimal_moves, visited):
        h = self.heuristic(env)

        predicted_cost = g_score + h

        if predicted_cost > threshold:
            return predicted_cost

        # if the current position is goal
        if h == 0:
            return -1

        min = np.inf

        for action in env.action_space:
            copied_game = env.copy()
            possible_env = gym.make(env_name)
            possible_env.reset()
            possible_env.setGame(copied_game)

            _, _, done, _ = possible_env.step(action)

            if done:
                h = self.heuristic(possible_env)

                # distance to flag is 0
                if h == 0:
                    optimal_moves.append(action)
                    return -1
                else:
                    return min

            game_state = self.get_env_game_state(possible_env)

            if game_state not in visited:
                visited.append(game_state)
                optimal_moves.append(action)

                temp = self.search(
                    possible_env, g_score + 1, threshold, optimal_moves, visited
                )

                if temp != -1:
                    optimal_moves.pop()
                    visited.pop()

                if temp < min:
                    min = temp
        return min

    def step(self, env: gym.Env):
        action = self.optimal_moves.pop(0)
        print(action)
        _, _, done, _ = env.step(action)
        return done

    def get_your_positions(self, env: gym.Env) -> List[np.array]:
        positions = env.game.GetMap().GetPositions(env.game.GetPlayerIcon())
        positions = [np.array(p) for p in positions]
        return positions

    def get_goal_positions(self, env: gym.Env) -> np.array:
        rule_manager = env.game.GetRuleManager()
        win_rules = rule_manager.GetRules(pyBaba.ObjectType.WIN)

        convert = pyBaba.ConvertTextToIcon
        win_positions = []
        game_map = env.game.GetMap()

        map_height = game_map.GetHeight()
        map_width = game_map.GetWidth()
        for y in range(map_height):
            for x in range(map_width):
                for win_rule in win_rules:

                    win_rule_type = win_rule.GetObjects()[0].GetTypes()[0]

                    if game_map.At(x, y).HasType(convert(win_rule_type)):
                        win_positions.append([x, y])

        return np.array(win_positions)


if __name__ == "__main__":
    register_baba_env(env_name, env_path)
    env = gym.make(env_name)
    env.reset()
    # state = env.reset().reshape(1, -1, 9, 11)
    moves = 40
    done = False
    agent = IDAStarAgent()

    start_time = time.time()
    agent.simulate(env)
    print(f"Total simulation time: {time.time() - start_time}s")

    while not done:
        done = agent.step(env)
        env.render()
        sleep(0.2)
