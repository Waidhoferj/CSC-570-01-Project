import random
import sys
from time import sleep

import pyBaba
import numpy as np
import gym
import random
from collections import defaultdict
from typing import List


import os
os.environ["SDL_AUDIODRIVER"] = "dsp"

def get_pos(rule_positions_cand):
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


def get_your_positions(env: gym.Env) -> List[np.array]:
    positions = env.game.GetMap().GetPositions(env.game.GetPlayerIcon())
    positions = [np.array(p) for p in positions]
    return positions

def get_goal_positions(env: gym.Env) -> np.array:
    rule_manager = env.game.GetRuleManager()
    win_rules = rule_manager.GetRules(pyBaba.ObjectType.WIN)

    convert = pyBaba.ConvertTextToIcon
    win_positions = []
    game_map = env.game.GetMap()

    for win_rule in win_rules:

        win_rule_type = win_rule.GetObjects()[0].GetTypes()[0]

        win_positions.extend(game_map.GetPositions(convert(win_rule_type)))

    return np.array(win_positions)

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

    return get_pos(rule_positions_cand)


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


def get_rules(env: gym.Env) -> np.array:
    num_rules = env.game.GetRuleManager().GetNumRules()
    rules = [env.game.GetRuleManager().GetPropertyRules()[idx] for idx in range(num_rules)]
    #print('rules', rules)

    rules_by_coords = defaultdict(None)
    rules_by_nouns = defaultdict(None)
    rules_by_verbs = defaultdict(None)
    rules_by_propt = defaultdict(None)
    # Store rules in a dict by NOUN (BABA, WALL, etc.) and by PROPERTY (YOU, WIN, etc.)

    for rule_idx in range(num_rules):
        rule_objects = rules[rule_idx].GetObjects()
        rule_positions_cand = [
            env.game.GetMap().GetPositions(rule_obj.GetTypes()[0])
            for rule_obj in rule_objects
        ]

        coords, rule_dir = get_pos(rule_positions_cand)
        if coords is None and rule_dir is None:
            continue

        rules_by_coords[tuple(coords)] = (rules[rule_idx], coords, rule_dir)
        for rule_obj in rule_objects:
            if rule_obj.HasNounType():
                #print('noun', rule_obj.GetTypes())
                rules_by_nouns[rule_obj.GetTypes()[0]] = (rules[rule_idx], coords, rule_dir)
            elif rule_obj.HasVerbType():
                #print('verb', rule_obj.GetTypes())
                rules_by_verbs[rule_obj.GetTypes()[0]] = (rules[rule_idx], coords, rule_dir)
            elif rule_obj.HasPropertyType():
                #print('property', rule_obj.GetTypes())
                rules_by_propt[rule_obj.GetTypes()[0]] = (rules[rule_idx], coords, rule_dir)
            else:
                print('Unknown type', rule_obj.GetTypes())

    #print('nouns', rules_by_nouns)
    #print('coords', rules_by_coords)
    return rules_by_coords, rules_by_nouns, rules_by_verbs, rules_by_propt


def get_features(env):
    rules_by_coords, rules_by_nouns, rules_by_verbs, rules_by_propt = get_rules(env)
    win_rule_exists = None
    win_reachable = None
    win_location = None
    #can_push_something = 0
    #pushable_object = 0
    #obstacles = 2d map
    #print('propt', type(rules_by_propt))
    height = env.game.GetMap().GetHeight()
    width = env.game.GetMap().GetWidth()
    #print('hw', height, width)

    if not rules_by_propt.get(pyBaba.ObjectType.WIN):
        #print('no win')
        win_rule_exists = 0
    else:
        win_rule_exists = 1


    goal_pos = None
    win_location = np.zeros((1, 7, 7))
    if win_rule_exists:
        goal_pos = get_goal_pos(env)
        goal_rows, goal_cols = goal_pos[:,1], goal_pos[:,0]
        win_location[0, goal_rows, goal_cols] = 1

    return win_location



    # state[0] - BABA_TEXT, TILE
    # state[1] - IS_TEXT
    # state[2] - YOU_TEXT
    # state[3] - ICON_EMPTY - walkable (not the only walkable tiles tho)
    # state[4] - FLAG_TEXT
    # state[5] - WIN_TEXT
    # state[6] - ICON_WALL (STOP ICONS) ???
    # state[7] - ICON_ROCK (PUSH ICONS) ???
    # state[8] - ICON_BABA
    # state[9] - ICON_FLAG
    # state[10] - WALL_TEXT
    # state[11] - STOP_TEXT
    # state[12] - ROCK_TEXT
    # state[13] - PUSH_TEXT
    # state[14] - TEXT?
    # state[15] - TEXT?

    # 1. We want IS YOU rules to see what the agent can move
         # How to store multiple moving objects?
    # 2. We want IS STOP  rules to get the obstacle map
    # 3. IS PUSH rules to find ways to move obstacles
    # 4. IS WIN rules to win ofc
         # Storing multiple WIN blocks shouldn't be a problem.

    # Provide features for the ML models
    # Features:
    # win rule exists = 1/0
    # win_reachable = 1/0
    # win_location = one hot 2d vector of map
    # can_push_something = 1/0
    # pushable_object = 2d map with 1s marking pushable objects
    # # this can be rocks, grass, or word icons (nouns, verbs, properties)
    # obstacles = 2d map showing reachable cells in the current state



    # Algorithm Mk1
    # Get WIN positions
        # IF no win conditions
            # try changing the WIN rule and continue
        # IF reachable
            # move to WIN
        # ELSE
            # try using some PUSH blocks to clear a way through the obstacles
        # try changing the WIN rule and continue



    # rule_st_is_you = env.game.GetRuleManager().GetRules(pyBaba.ObjectType.IS)[0]  # returns [BABA, IS, YOU]
    # print('rule_st_is_you', rule_st_is_you)
    # rule_objs = rule_st_is_you.GetObjects()
    # for rule in rule_objs:
    #     print('rule_objs', rule.GetTypes()[0])
    # noun = rule_objs[0].GetTypes()[0]
    # noun_pos = env.game.GetMap().GetPositions(noun)
    # print('noun', noun)
    # rule_positions_cand = [
    #     env.game.GetMap().GetPositions(rule_obj.GetTypes()[0]) for rule_obj in rule_objs
    # ]
    # print('rule_positions_cand', rule_positions_cand)
