import copy
from typing import List
import gym
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

import pyBaba
import pygame
import rendering
from gym import spaces
import itertools

from nlp_heuristic import get_features


# Registration check
registered_envs = set()


def register_baba_env(
    name: str, max_episode_steps=200, env_class_str="BabaEnv", **kwargs
):
    if name not in registered_envs:
        registered_envs.add(name)
        register(
            id=name,
            entry_point=f"environment:{env_class_str}",
            max_episode_steps=max_episode_steps,
            nondeterministic=True,
            kwargs=kwargs,
        )


class BabaEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, path: str = "", enable_render=True, user_controls=False, extra_features=None):
        super(BabaEnv, self).__init__()
        self.path = path
        self.game = pyBaba.Game(self.path)
        self.enable_render = enable_render
        self.extra_features = extra_features
        if enable_render:
            self.renderer = rendering.Renderer(self.game)

        if user_controls and enable_render:
            self.setup_user_controls()

        self.action_space = spaces.Discrete(4)

        self.action_space = [
            pyBaba.Direction.UP,
            pyBaba.Direction.DOWN,
            pyBaba.Direction.LEFT,
            pyBaba.Direction.RIGHT,
        ]
        self.observation_space = spaces.MultiBinary(self.get_obs().shape)

        self.action_size = len(self.action_space)
        self.immovable_rule_objs = self.get_immovable_rule_objs()

        self.seed()
        self.reset()

    def setup_user_controls(self):
        done = False

        def handle_movement(event):
            action = None
            nonlocal done
            if event.unicode == "w":
                action = pyBaba.Direction.UP
            elif event.unicode == "a":
                action = pyBaba.Direction.LEFT
            elif event.unicode == "s":
                action = pyBaba.Direction.DOWN
            elif event.unicode == "d":
                action = pyBaba.Direction.RIGHT
            _, _, done, _ = self.step(action)

            if done:
                self.reset()

        self.renderer.on(pygame.KEYUP, handle_movement)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        self.game.Reset()
        self.done = False

        return self.get_obs()

    def step(self, action):
        action = (
            action if type(action) == pyBaba.Direction else self.action_space[action]
        )
        self.game.MovePlayer(action)

        result = self.game.GetPlayState()

        if result == pyBaba.PlayState.LOST:
            self.done = True
            reward = -100
        elif result == pyBaba.PlayState.WON:
            self.done = True
            reward = 200
        else:
            reward = -0.5

        return self.get_obs(), reward, self.done, {}
    
    def solved_step(self, action, end_filepath='levels/out/x_end.txt'):
        """ special step function for solution agent """
        action = (
            action if type(action) == pyBaba.Direction else self.action_space[action]
        )
        self.game.MovePlayer(action)

        result = self.game.GetPlayState()

        if result == pyBaba.PlayState.LOST:
            self.done = True
            reward = -100
        elif result == pyBaba.PlayState.WON:
            self.done = True
            reward = 200
            self.game.GetMap().Write(end_filepath)
        else:
            reward = -0.5

        return self.get_obs(), reward, self.done, {}

    def render(self, mode="human", close=False):
        if not self.enable_render:
            raise Exception("Renderer not enabled")
        if close:
            self.renderer.quit_game()

        return self.renderer.render(self.game.GetMap(), mode)

    def copy(self):
        return copy.deepcopy(self.game)

    def set_game(self, game):
        self.game = game

    def get_obs(self):
        game_tensor = np.array(
            pyBaba.Preprocess.StateToTensor(self.game), dtype=np.float32
        ).reshape(-1, self.game.GetMap().GetHeight(), self.game.GetMap().GetWidth())
        #print('game', game_tensor.shape)
        if self.extra_features is None:
           return game_tensor
        else:
            tmp = []
            if 'WIN' in self.extra_features:
                tmp.append(get_features(self))
                #print('WIN', tmp[0].shape)
            if 'PROPT' in self.extra_features:
                tmp.append(get_property_positions(self))
                #print('PROPT', tmp[0].shape)
            if len(tmp) > 1:
                #print(tmp[0].shape, tmp[1].shape)
                tmp = np.concatenate((tmp[0], tmp[1]), axis=0)
                #print(tmp.shape)
                return np.concatenate((game_tensor, tmp), axis=0)
            else:
                return np.concatenate((game_tensor, tmp[0]), axis=0)

    def get_immovable_rule_objs(self):
        # get rules that are on the corders
        m_width = self.game.GetMap().GetWidth()
        m_height = self.game.GetMap().GetHeight()

        immovable_objs = {}

        for rule in self.game.GetRuleManager().GetPropertyRules():
            rule_objs = rule.GetObjects()

            rule_pos, _ = self.get_rule_pos_direction(rule_objs)

            noun_obj, op_obj, prop_obj = rule_objs

            noun_obj = noun_obj.GetTypes()[0]
            op_obj = op_obj.GetTypes()[0]
            prop_obj = prop_obj.GetTypes()[0]

            print("immovable_obj")
            print(noun_obj, op_obj, prop_obj)
            noun_obj_pos, op_obj_pos, prop_obj_pos = rule_pos

            is_at_corners = False

            # if the object is part of st_is_you, then can't move
            st_is_you_pos, _ = self.get_rule_w_property(pyBaba.ObjectType.YOU)

            if [noun_obj_pos, op_obj_pos, prop_obj_pos] == st_is_you_pos:
                if noun_obj in immovable_objs:
                    immovable_objs[noun_obj].add(noun_obj_pos)
                else:
                    immovable_objs[noun_obj] = {noun_obj_pos}

                if op_obj in immovable_objs:
                    immovable_objs[op_obj].add(op_obj_pos)
                else:
                    immovable_objs[op_obj] = {op_obj_pos}

                if prop_obj in immovable_objs:
                    immovable_objs[prop_obj].add(prop_obj_pos)
                else:
                    immovable_objs[prop_obj] = {prop_obj_pos}

            # if nouns at corners
            if (
                (0, 0) == noun_obj_pos
                or (0, m_height - 1) == noun_obj_pos
                or (m_width - 1, 0) == noun_obj_pos
                or (m_width - 1, m_height - 1) == noun_obj_pos
            ):
                is_at_corners = True
                if noun_obj in immovable_objs:
                    immovable_objs[noun_obj].add(noun_obj_pos)
                else:
                    immovable_objs[noun_obj] = {noun_obj_pos}

                if op_obj in immovable_objs:
                    immovable_objs[op_obj].add(op_obj_pos)
                else:
                    immovable_objs[op_obj] = {op_obj_pos}

                if prop_obj in immovable_objs:
                    immovable_objs[prop_obj].add(prop_obj_pos)
                else:
                    immovable_objs[prop_obj] = {prop_obj_pos}

            # if props at corners
            if (
                (0, 0) == prop_obj_pos
                or (0, m_height - 1) == prop_obj_pos
                or (m_width - 1, 0) == prop_obj_pos
                or (m_width - 1, m_height - 1) == prop_obj_pos
            ):
                is_at_corners = True
                if noun_obj in immovable_objs:
                    immovable_objs[noun_obj].add(noun_obj_pos)
                else:
                    immovable_objs[noun_obj] = {noun_obj_pos}

                if op_obj in immovable_objs:
                    immovable_objs[op_obj].add(op_obj_pos)
                else:
                    immovable_objs[op_obj] = {op_obj_pos}

                if prop_obj in immovable_objs:
                    immovable_objs[prop_obj].add(prop_obj_pos)
                else:
                    immovable_objs[prop_obj] = {prop_obj_pos}

            if not is_at_corners:
                # Add to the object if all three can't move
                # check if obj can move
                if not self.can_move_obj(noun_obj_pos):
                    if noun_obj in immovable_objs:
                        immovable_objs[noun_obj].add(noun_obj_pos)
                    else:
                        immovable_objs[noun_obj] = {noun_obj_pos}

                if not self.can_move_obj(op_obj_pos):
                    if op_obj in immovable_objs:
                        immovable_objs[op_obj].add(op_obj_pos)
                    else:
                        immovable_objs[op_obj] = {op_obj_pos}

                if not self.can_move_obj(prop_obj_pos):
                    if prop_obj in immovable_objs:
                        immovable_objs[prop_obj].add(prop_obj_pos)
                    else:
                        immovable_objs[prop_obj] = {prop_obj_pos}
            print(immovable_objs)
            # TODO: add it to get usable objects
            # TODO: when forming a new rule, filter out those not movable
        return immovable_objs

    def update_immovable_objs(self):
        # update immovable objs
        print("there")
        self.immovable_rule_objs = self.get_immovable_rule_objs()

    def can_move_obj(self, obj_pos):
        x, y = obj_pos
        m_width = self.game.GetMap().GetWidth()
        m_height = self.game.GetMap().GetHeight()

        can_move_left = False
        can_move_right = False
        can_move_up = False
        can_move_down = False

        # right-closest empty space
        right_closest_empty_x = self.get_empty_space_pos(
            obj_pos, pyBaba.Direction.RIGHT
        )

        # left-closest empty space
        left_closest_empty_x = self.get_empty_space_pos(obj_pos, pyBaba.Direction.LEFT)

        # up-closest empty space
        up_closest_empty_y = self.get_empty_space_pos(obj_pos, pyBaba.Direction.UP)

        # down-closest empty space
        down_closest_empty_y = self.get_empty_space_pos(obj_pos, pyBaba.Direction.DOWN)

        # check if it's text or pushable object
        if right_closest_empty_x < m_width:
            can_move_left = self.game.CanMove(
                right_closest_empty_x, y, pyBaba.Direction.LEFT
            )

        if left_closest_empty_x >= 0:
            can_move_right = self.game.CanMove(
                left_closest_empty_x, y, pyBaba.Direction.RIGHT
            )

        if down_closest_empty_y < m_height:
            can_move_up = self.game.CanMove(
                x, down_closest_empty_y, pyBaba.Direction.UP
            )

        if up_closest_empty_y >= 0:
            can_move_down = self.game.CanMove(
                x, up_closest_empty_y, pyBaba.Direction.DOWN
            )

        # print(x, y, can_move_left, can_move_right, can_move_up, can_move_down)

        return can_move_left or can_move_right or can_move_up or can_move_down

    def get_empty_space_pos(self, start_pos, dir):
        start_x, start_y = start_pos
        m_width = self.game.GetMap().GetWidth()
        m_height = self.game.GetMap().GetHeight()

        if dir == pyBaba.Direction.LEFT:
            # can't push
            if start_x == 0:
                return -1

            curr_x = start_x - 1
            while curr_x >= 0 and not self.game.GetMap().At(curr_x, start_y).HasType(
                pyBaba.ICON_EMPTY
            ):
                curr_x -= 1
            return curr_x

        elif dir == pyBaba.Direction.RIGHT:
            # can't push
            if start_x == m_width - 1:
                return m_width

            curr_x = start_x + 1
            while curr_x < m_width and not self.game.GetMap().At(
                curr_x, start_y
            ).HasType(pyBaba.ICON_EMPTY):
                curr_x += 1
            return curr_x

        elif dir == pyBaba.Direction.UP:
            # can't push
            if start_y == 0:
                return -1

            curr_y = start_y - 1
            while curr_y >= 0 and not self.game.GetMap().At(start_x, curr_y).HasType(
                pyBaba.ICON_EMPTY
            ):
                curr_y -= 1
            return curr_y

        elif dir == pyBaba.Direction.DOWN:
            # can't push
            if start_y == m_height - 1:
                return m_height

            curr_y = start_y + 1
            while curr_y < m_height and not self.game.GetMap().At(
                start_x, curr_y
            ).HasType(pyBaba.ICON_EMPTY):
                curr_y += 1
            return curr_y

    # get_direction of rules
    def get_rule_pos_direction(self, rule_objs):
        rule_positions_cand = [
            self.game.GetMap().GetPositions(rule_obj.GetTypes()[0])
            for rule_obj in rule_objs
        ]

        prop_positions = rule_positions_cand[-1]
        for prop_position in prop_positions:
            # check left to right
            op_pos = (prop_position[0] - 1, prop_position[1])

            if op_pos in rule_positions_cand[1]:
                obj_pos = (op_pos[0] - 1, op_pos[1])
                if obj_pos in rule_positions_cand[0]:
                    return (
                        [obj_pos, op_pos, prop_position],
                        pyBaba.RuleDirection.HORIZONTAL,
                    )

            # check top to bottom
            op_pos = (prop_position[0], prop_position[1] - 1)

            if op_pos in rule_positions_cand[1]:
                obj_pos = (op_pos[0], op_pos[1] - 1)
                if obj_pos in rule_positions_cand[0]:
                    return (
                        [obj_pos, op_pos, prop_position],
                        pyBaba.RuleDirection.VERTICAL,
                    )

        return None, None

    def get_rule_w_property(self, prop_type):
        rule_st_is_you = self.game.GetRuleManager().GetRules(prop_type)[
            0
        ]  # ex) [BABA, IS, YOU]
        rule_objs = rule_st_is_you.GetObjects()

        if len(self.game.GetMap().GetPositions(rule_objs[-1].GetTypes()[0])) > 1:
            return None, None

        return self.get_rule_pos_direction(rule_objs)

    def get_movable_objs(self):
        # get noun types, op types, and property types
        noun_types_pos = {}
        op_types_pos = {}
        prop_types_pos = {}
        for obj in self.game.GetMap().GetObjects():
            if obj.HasNounType():
                noun_type = obj.GetTypes()[0]
                if noun_type not in noun_types_pos:
                    noun_pos = self.game.GetMap().GetPositions(noun_type)

                    # filter those in the corner
                    noun_pos = list(
                        filter(
                            lambda pos: (noun_type not in self.immovable_rule_objs)
                            or (
                                noun_type in self.immovable_rule_objs
                                and pos not in self.immovable_rule_objs[noun_type]
                            ),
                            noun_pos,
                        )
                    )

                    if len(noun_pos):
                        noun_types_pos[noun_type] = noun_pos

            elif obj.HasVerbType():
                verb_type = obj.GetTypes()[0]
                if verb_type not in op_types_pos:
                    verb_pos = self.game.GetMap().GetPositions(verb_type)

                    # filter those in the corner
                    verb_pos = list(
                        filter(
                            lambda pos: (verb_type not in self.immovable_rule_objs)
                            or (
                                verb_type in self.immovable_rule_objs
                                and pos not in self.immovable_rule_objs[verb_type]
                            ),
                            verb_pos,
                        )
                    )

                    if len(verb_pos):
                        op_types_pos[verb_type] = verb_pos

            elif obj.HasPropertyType():
                prop_type = obj.GetTypes()[0]
                if prop_type not in prop_types_pos:
                    prop_pos = self.game.GetMap().GetPositions(prop_type)

                    # filter those in the corner
                    prop_pos = list(
                        filter(
                            lambda pos: (prop_type not in self.immovable_rule_objs)
                            or (
                                prop_type in self.immovable_rule_objs
                                and pos not in self.immovable_rule_objs[prop_type]
                            ),
                            prop_pos,
                        )
                    )

                    if len(prop_pos):
                        prop_types_pos[prop_type] = prop_pos
        return noun_types_pos, op_types_pos, prop_types_pos

    def get_usable_objs(self):
        (
            movable_noun_types_pos,
            movable_op_types_pos,
            movable_prop_types_pos,
        ) = self.get_movable_objs()

        noun_types_pos = movable_noun_types_pos.copy()
        op_types_pos = movable_op_types_pos.copy()
        prop_types_pos = movable_prop_types_pos.copy()

        for rule in self.game.GetRuleManager().GetPropertyRules():
            rule_objs = rule.GetObjects()

            rule_pos, _ = self.get_rule_pos_direction(rule_objs)

            noun_obj, op_obj, prop_obj = rule_objs

            noun_obj = noun_obj.GetTypes()[0]
            op_obj = op_obj.GetTypes()[0]
            prop_obj = prop_obj.GetTypes()[0]

            # print(noun_obj, op_obj, prop_obj)

            noun_obj_pos, op_obj_pos, prop_obj_pos = rule_pos

            if (
                noun_obj in movable_noun_types_pos
                or op_obj in movable_op_types_pos
                or prop_obj in movable_prop_types_pos
            ):
                if noun_obj in movable_noun_types_pos:
                    noun_types_pos[noun_obj].append(noun_obj_pos)
                else:
                    noun_types_pos[noun_obj] = [noun_obj_pos]

                if op_obj in movable_op_types_pos:
                    op_types_pos[op_obj].append(op_obj_pos)
                else:
                    op_types_pos[op_obj] = [op_obj_pos]

                if prop_obj in movable_prop_types_pos:
                    prop_types_pos[prop_obj].append(prop_obj_pos)
                else:
                    prop_types_pos[prop_obj] = [prop_obj_pos]

        return noun_types_pos, op_types_pos, prop_types_pos

    def win_rule_exists(self):
        win_rule_exist = False
        for rule in self.game.GetRuleManager().GetPropertyRules():
            rule_objs = rule.GetObjects()
            for rule_obj in rule_objs:
                win_rule = rule_obj.GetTypes()[0]
                if win_rule == pyBaba.WIN:
                    win_rule_exist = True
        return win_rule_exist

    def get_curr_rules(self):
        curr_rules = set()
        for rule in self.game.GetRuleManager().GetPropertyRules():
            curr_rule = []
            for obj in rule.GetObjects():
                curr_rule.append(obj.GetTypes()[0])
            curr_rules.add(tuple(curr_rule))

        return curr_rules

    def get_new_rules(self):
        noun_types_pos, op_types_pos, prop_types_pos = self.get_usable_objs()

        print("noun", noun_types_pos)
        print("op", op_types_pos)
        print("prop", prop_types_pos)

        noun_types = list(noun_types_pos.keys())
        op_types = list(op_types_pos.keys())
        prop_types = list(prop_types_pos.keys())
        curr_rules = self.get_curr_rules()

        cand_rules = []
        for comb in itertools.product(noun_types, op_types, prop_types):
            if comb not in curr_rules:
                cand_rules.append(comb)

        return cand_rules

    def get_obj_positions(self, obj_type):
        noun_types_pos, op_types_pos, prop_types_pos = self.get_usable_objs()

        print("get_obj_pos")
        print(noun_types_pos)
        print(op_types_pos)
        print(prop_types_pos)

        if obj_type in noun_types_pos:
            return noun_types_pos[obj_type]
        elif obj_type in op_types_pos:
            return op_types_pos[obj_type]
        elif obj_type in prop_types_pos:
            return prop_types_pos[obj_type]
        else:
            return []

class PropertyBasedEnv(BabaEnv):
    def get_obs(self):
        return get_property_positions(self, stage_size=(30,30))

class PaddedObsEnv(BabaEnv):
    def get_obs(self, stage_size=(20,20)):
            obs = super().get_obs()
            masks, old_h, old_w = obs.shape
            padded_obs = np.zeros((masks, *stage_size))
            padded_obs[:,:old_h, :old_w] = obs
            return padded_obs


class ProgressiveTrainingEnv(PropertyBasedEnv):
    def __init__(self, levels:List[str], 
    enable_render=True, 
    score_threshold=170, 
    episode_range=[20, 20], 
    patience=3,
    max_stage_size = (30,30)):
        assert len(levels) > 0
        super().__init__(levels[0], enable_render, user_controls=False)
        self.levels = levels
        self.score_history = []
        self.score_threshold = score_threshold
        self.episode_range = episode_range
        self.patience = patience
        self.episode = 0
        self.wins = 0
        self.level_idx = 0
        self.score = 0
        self.level_history = {}
        self.level_changed = True
        self.max_stage_size = max_stage_size
        self.observation_space = spaces.MultiBinary(self.get_obs().shape)
        

    def next_level(self):
        self.level_history[self.levels[self.level_idx]] = {"episodes": self.episode, "score": self.avg_score(), "wins": self.wins}
        print(self.level_history)
        self.level_changed = True
        self.score_history = []
        self.episode = 0
        self.wins = 0
        self.level_idx +=1
        if not self.levels_complete():
            super().__init__(self.levels[self.level_idx], self.enable_render, False)
            print("ProgressiveTrainingEnv: Moving to level ", self.levels[self.level_idx])


    def avg_score(self,window_size=100):
        return np.mean(self.score_history[-window_size:])
    
    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        self.score += reward
        self.level_changed = True
        if done:
            self.episode +=1
            self.wins +=1
            self.score_history.append(self.score)
            self.score = 0
            h_size = len(self.score_history)
            # out_of_patience =h_size >= self.patience and all(self.score_history[i] >= self.score_history[i+1] for i in range(h_size - self.patience, h_size -1))
            # mastered_level = self.avg_score() > self.score_threshold and self.episode >= self.episode_range[0]
            out_of_episodes = self.episode_range[1] < self.episode

            if out_of_episodes:
                self.next_level()
        return next_obs, reward, done, info
    
    def levels_complete(self) -> bool:
        return len(self.levels) <= self.level_idx

    def get_report(self):
        return self.level_history


def get_property_positions(env: gym.Env, stage_size=None) -> np.array:
    """
    Finds the position of every object with selected properties on the map.
    Encodes this into a numpy bitmask with shape (map_width, map_height, n_properties)
    """
    properties = [
        "YOU",
        "STOP",
        "PULL",
        "PUSH",
        "SWAP",
        # "TELE",
        "MOVE",
        # "FALL",
        # "SHIFT",
        "WIN",
        "DEFEAT",
        "SINK",
        "HOT",
        "MELT",
        "SHUT",
        "OPEN",
        "WEAK",
        "FLOAT",
        # "MORE",
        # "UP",
        # "DOWN",
        # "LEFT",
        # "RIGHT",
        "WORD",
        # "BEST",
        # "SLEEP",
        # "RED",
        # "BLUE",
        # "HIDE",
        # "BONUS",
        # "END",
        # "DONE",
        # "SAFE",
    ]

    properties = [getattr(pyBaba.ObjectType, prop) for prop in properties]
    game_map = env.game.GetMap()
    rule_manager = env.game.GetRuleManager()
    property_positions = []
    map_width, map_height = (game_map.GetWidth(), game_map.GetHeight())
    width, height = (map_width, map_height) if stage_size is None else stage_size

    if stage_size is not None:
        reachable_mask = np.zeros((height,width))
        reachable_mask[:map_height, :map_width] = 1
        property_positions.append(reachable_mask)

    
    for property in properties:
        rules = rule_manager.GetRules(property)

        convert = pyBaba.ConvertTextToIcon
        positions = np.zeros((height, width))
        for rule in rules:
            obj_type = rule.GetObjects()[0].GetTypes()[0]
            for pos in game_map.GetPositions(convert(obj_type)):
                positions[pos[1]][pos[0]] = 1
        property_positions.append(positions)

    # Find all text locations
    text_mask = np.zeros((height, width))
    for y in range(map_height):
        for x in range(map_width):
            if game_map.At(x,y).HasTextType():
                text_mask[y,x] = 1
    property_positions.append(text_mask)



    # Push is a special case since all text should be push as well
    push_idx = properties.index(pyBaba.ObjectType.PUSH)
    property_positions[push_idx][text_mask == 1] = 1
    

    return np.stack(property_positions)



