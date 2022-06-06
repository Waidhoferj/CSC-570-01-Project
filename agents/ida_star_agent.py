from collections import defaultdict
import os
from time import sleep

env_name = "baba-babaisyou-v0"
env_path = os.path.join("baba-is-auto", "Resources", "Maps", "baba_is_you.txt")

env_name = "baba-level-v0"
env_path = os.path.join("levels", "out", "3.txt")

from environment import register_baba_env
from rule_utils import create_win_rule
from utils import is_breaking_rule, train_test_levels

import pdb
import pyBaba
import numpy as np
import gym
import random
from typing import List
import time
import json


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

        new_rules = env.get_new_rules()

        # for rule in new_rules:
        #     print(rule)

        # create win rule if you don't have
        if not env.win_rule_exists():
            can_create_win_rule = create_win_rule(
                env, new_rules, self, enable_render=True
            )
            if not can_create_win_rule:
                print("Not solvable!")
                exit(-1)

        threshold = self.heuristic(env)

        moves_taken = 0
        optimal_moves = []

        while True:
            possible_env = gym.make(env_name)
            possible_env.reset()

            new_rules = possible_env.get_new_rules()

            # for rule in new_rules:
            #     print(rule)

            # create win rule if you don't have
            if not possible_env.win_rule_exists():
                can_create_win_rule = create_win_rule(
                    possible_env, new_rules, self, enable_render=True
                )
                if not can_create_win_rule:
                    print("Not solvable!")
                    exit(-1)

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
        rule_positions, rule_direction = env.get_rule_w_property(pyBaba.ObjectType.YOU)

        if not rule_positions:
            return True
        else:
            return not is_breaking_rule(
                action, positions, rule_positions, rule_direction
            )

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
                prev_game = env.copy()

                curr_obs, _, done, _ = env.step(action)

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
                        print("lost")
                        return np.inf

                state = tuple(curr_obs.reshape(-1).tolist())

                if state not in visited:
                    visited.add(state)
                    optimal_moves.append(action)

                    env.render()

                    temp = self.search(
                        env, g_score + 1, threshold, optimal_moves, visited
                    )

                    # found
                    if temp == -1:
                        return temp

                    if temp < min_cost:
                        min_cost = temp

                    optimal_moves.pop()

                if is_moved:
                    env.set_game(prev_game)
                    env.render()

        return min_cost

    def step(self, env: gym.Env):
        action = self.optimal_moves.pop(0)
        print(action)
        _, r, done, _ = env.step(action)
        return r, done

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
    train, test = train_test_levels()
    levels =[*train,*test]
    level_performance = {}
    for i,level in enumerate(levels):
        env_name = f"baba-babaisyou{i}-v0"
        register_baba_env(env_name, path=level, max_episode_steps=250)
        env = gym.make(env_name)
        env.reset()
        agent = IDAStarAgent()

        start_time = time.time()
        level_performance[level] = {"score": 0, "steps": 0, "won": False}
        try:
            agent.simulate(env)
            score = 0
            steps = 0
            reward = 0
            done = False
            while not done:
                reward, done = agent.step(env)
                score += reward
                steps +=1
                env.render()
                sleep(0.2)
            level_performance[level] = {"score": score, "steps": steps, "won": reward > 0}
        except:
            continue
    with open(f"{os.path.split(__file__)[0]}/../Results/ida_star_results.json", "w") as f:
        json.dump(level_performance,f)

