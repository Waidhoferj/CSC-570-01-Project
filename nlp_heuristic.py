import random
import sys
from time import sleep


from environment import register_baba_env

import rendering

import pyBaba
import numpy as np
import gym
import random

import os

os.environ["SDL_AUDIODRIVER"] = "dsp"

# Function to get positions of movable blocks
def get_push_pos(env: gym.Env) -> np.array:
    pass


# Function to get rule connected to YOU (from idaq_star_agent code)
def get_is_you_pos(env):
    rule_st_is_you = env.game.GetRuleManager().GetRules(pyBaba.ObjectType.YOU)[
        0
    ]  # returns [BABA, IS, YOU]
    rule_objs = rule_st_is_you.GetObjects()

    rule_positions_cand = [
        env.game.GetMap().GetPositions(rule_obj.GetTypes()[0]) for rule_obj in rule_objs
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


# Function to get rule WIN (from idaq_star_agent code)
def get_goal_pos(env: gym.Env) -> np.array:
    rule_manager = env.game.GetRuleManager()
    win_rules = rule_manager.GetRules(pyBaba.ObjectType.WIN)

    convert = pyBaba.ConvertTextToIcon
    win_positions = []
    game_map = env.game.GetMap()

    for win_rule in win_rules:

        win_rule_type = win_rule.GetObjects()[0].GetTypes()[0]

        win_positions.extend(game_map.GetPositions(convert(win_rule_type)))

    return np.array(win_positions)
