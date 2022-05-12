import random
import sys
from time import sleep

env_name = "baba-volcano-v0"
sys.path.append("../baba-is-auto/Extensions/BabaRL/" + env_name)
import environment
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


class AStarAgent:
    def __init__(self):
        self.moves = []  # the final set of most optimal moves
        self.frontier = [
            ()
        ]  # Priority queue of what move to take next (predicted_cost, env, moves_taken)
        self.best_solution = (np.inf, [])

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

    def backtrack(self, win_state):
        moves = []
        state = win_state
        while state is not None:
            parent = state[5]
            action = state[4]
            moves.append(action)
            state = parent
        # ignore the first element: None added in the beginning of simulation
        return list(reversed(moves))[1:]

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
        done = False
        # to break the tie of h
        counter = 0
        heuristic = self.heuristic(env)
        # predicted_cost, moves_taken, counter, environment, action, parent
        self.frontier = [(heuristic, 0, counter, env, None, None)]

        visited = []

        while len(self.frontier) > 0:
            state = heapq.heappop(self.frontier)
            predicted_cost, moves_taken, _, env, _, _ = state

            # Check if we have already visited
            env_game_state = self.get_env_game_state(env)

            if env_game_state in visited:
                continue

            visited.append(env_game_state)

            moves_taken += 1
            for action in env.action_space:
                counter += 1

                copied_game = env.copy()
                possible_env = gym.make(env_name)
                possible_env.reset()
                possible_env.setGame(copied_game)

                _, _, done, _ = possible_env.step(action)

                h = self.heuristic(possible_env)

                predicted_cost = h + moves_taken

                # Prune useless paths
                if predicted_cost > self.best_solution[0]:
                    continue

                entry = (
                    predicted_cost,
                    moves_taken,
                    counter,
                    possible_env,
                    action,
                    state,
                )

                if done:
                    # if we found the goal
                    if h == 0:
                        cost = predicted_cost
                        prev_cost = self.best_solution[0]
                        if cost < prev_cost:
                            actions = self.backtrack(entry)
                            self.best_solution = (cost, actions)
                    continue

                heapq.heappush(self.frontier, entry)

        print(self.best_solution)

    def step(self, env: gym.Env):
        action = self.best_solution[1].pop(0)
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
    env = gym.make(env_name)
    state = env.reset().reshape(1, -1, 9, 11)
    moves = 40
    done = False
    agent = AStarAgent()

    start_time = time.time()
    agent.simulate(env)
    print(f"Total simulation time: {time.time() - start_time}s")

    while not done:
        done = agent.step(env)
        env.render()
        sleep(0.2)
