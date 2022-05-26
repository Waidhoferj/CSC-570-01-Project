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
        optimal_moves = []
        i = 0

        while True:
            possible_env = gym.make(env_name)
            possible_env.reset()

            visited = {tuple(possible_env.get_obs().reshape(-1).tolist())}

            temp = self.search(
                possible_env, moves_taken, threshold, optimal_moves, visited
            )

            # when the goal is found
            if temp == -1:
                self.optimal_moves = optimal_moves
                print(self.optimal_moves)
                return

            # can't find the optimal moves
            if temp == np.inf:
                return

            threshold = temp

    def can_move(self, env, action) -> bool:
        """
        Check if the action breaks a rule
        """

        # cost of breaking a property rule
        positions = self.get_your_positions(env)

        # rule: _ is YOU
        rule_positions, rule_direction = self.get_st_is_you_positions(env)

        if not rule_positions:
            return True
        else:
            return not self.is_breaking_rule(
                action, positions, rule_positions, rule_direction
            )

    def is_breaking_rule(self, action, your_positions, rule_positions, rule_direction):
        your_x_pos, your_y_pos = your_positions[0]

        if rule_direction == pyBaba.RuleDirection.HORIZONTAL:
            if action == pyBaba.Direction.UP:
                return (your_x_pos, your_y_pos - 1) in rule_positions
            elif action == pyBaba.Direction.DOWN:
                return (your_x_pos, your_y_pos + 1) in rule_positions
            else:
                return False
        elif rule_direction == pyBaba.RuleDirection.VERTICAL:
            if action == pyBaba.Direction.LEFT:
                return (your_x_pos - 1, your_y_pos) in rule_positions
            elif action == pyBaba.Direction.RIGHT:
                return (your_x_pos + 1, your_y_pos) in rule_positions
            else:
                return False
        else:
            print("Unrecognized rule direction!")
            exit(-1)

    def get_st_is_you_positions(self, env):
        rule_st_is_you = env.game.GetRuleManager().GetRules(pyBaba.ObjectType.YOU)[
            0
        ]  # returns [BABA, IS, YOU]
        rule_objs = rule_st_is_you.GetObjects()

        rule_positions_cand = [
            env.game.GetMap().GetPositions(rule_obj.GetTypes()[0])
            for rule_obj in rule_objs
        ]

        if len(rule_positions_cand[-1]) > 1:
            return None, None

        # check left to right
        you_pos = rule_positions_cand[-1][0]
        is_pos = (you_pos[0] - 1, you_pos[1])

        if is_pos in rule_positions_cand[1]:
            obj_pos = (is_pos[0] - 1, is_pos[1])
            if obj_pos in rule_positions_cand[0]:
                return [obj_pos, is_pos, you_pos], pyBaba.RuleDirection.HORIZONTAL

        # check top to bottom
        is_pos = (you_pos[0], you_pos[1] - 1)

        if is_pos in rule_positions_cand[1]:
            obj_pos = (is_pos[0], is_pos[1] - 1)
            if obj_pos in rule_positions_cand[0]:
                return [obj_pos, is_pos, you_pos], pyBaba.RuleDirection.VERTICAL

        return None, None

    def search(self, env, g_score, threshold, optimal_moves, visited):
        h = self.heuristic(env)

        predicted_cost = g_score + h

        if predicted_cost > threshold:
            return predicted_cost

        # if the current position is goal
        if env.game.GetPlayState() == pyBaba.PlayState.WON:
            return -1

        min_cost = np.inf

        for action in env.action_space:
            if self.can_move(env, action):
                prev = env.get_obs()
                curr_obs, _, done, _ = env.env.step(action)

                # check in case the agent didn't move because it's blocked
                is_moved = (prev != curr_obs).any()

                if done:

                    playState = env.game.GetPlayState()

                    # win
                    if playState == pyBaba.PlayState.WON:
                        optimal_moves.append(action)
                        return -1

                    # when the timelimit is signalled, but game is still playing
                    elif playState == pyBaba.PlayState.PLAYING:
                        return min_cost

                    # lose
                    else:
                        return np.inf

                # game_state = self.get_env_game_state(env)
                state = tuple(curr_obs.reshape(-1).tolist())

                if state not in visited:
                    visited.add(state)
                    optimal_moves.append(action)

                    # env.render()

                    temp = self.search(
                        env, g_score + 1, threshold, optimal_moves, visited
                    )

                    if temp == -1:
                        return temp

                    if temp < min_cost:
                        min_cost = temp

                    optimal_moves.pop()

                if is_moved:
                    # backtrack

                    back_action = self.get_back_action(action)
                    env.step(back_action)

        return min_cost

    def get_back_action(self, action: pyBaba.Direction):
        if action == pyBaba.Direction.UP:
            return pyBaba.Direction.DOWN

        elif action == pyBaba.Direction.DOWN:
            return pyBaba.Direction.UP

        elif action == pyBaba.Direction.LEFT:
            return pyBaba.Direction.RIGHT

        elif action == pyBaba.Direction.RIGHT:
            return pyBaba.Direction.LEFT

        else:
            print("Unrecognized action during backtrack!")
            exit(-1)

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

        for win_rule in win_rules:

            win_rule_type = win_rule.GetObjects()[0].GetTypes()[0]

            win_positions.extend(game_map.GetPositions(convert(win_rule_type)))

        return np.array(win_positions)


if __name__ == "__main__":
    register_baba_env(env_name, env_path, enable_render=False)
    env = gym.make(env_name)
    env.reset()

    moves = 40
    done = False
    agent = IDAStarAgent()

    start_time = time.time()
    agent.simulate(env)
    print(f"Total simulation time: {time.time() - start_time}s")

    while not done:
        done = agent.step(env)
        # env.render()
        sleep(0.2)
