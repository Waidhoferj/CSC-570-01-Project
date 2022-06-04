import copy
import gym
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

import pyBaba
import pygame
import rendering
from gym import spaces
import itertools

# Registration check
registered_envs = set()


def register_baba_env(
    name: str, path: str, enable_render=True, max_episode_steps=200, user_controls=False
):
    if name not in registered_envs:
        registered_envs.add(name)
        register(
            id=name,
            entry_point="environment:BabaEnv",
            max_episode_steps=max_episode_steps,
            nondeterministic=True,
            kwargs={
                "path": path,
                "enable_render": enable_render,
                "user_controls": user_controls,
            },
        )


class BabaEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, path: str = "", enable_render=True, user_controls=False):
        super(BabaEnv, self).__init__()
        self.path = path
        self.game = pyBaba.Game(self.path)
        self.enable_render = enable_render

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
        return np.array(
            pyBaba.Preprocess.StateToTensor(self.game), dtype=np.float32
        ).reshape(-1, self.game.GetMap().GetHeight(), self.game.GetMap().GetWidth())

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
        noun_types_pos, op_types_pos, prop_types_pos = self.get_movable_objs()

        print(noun_types_pos, op_types_pos, prop_types_pos)

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
        noun_types_pos, op_types_pos, prop_types_pos = self.get_movable_objs()

        if obj_type in noun_types_pos:
            return noun_types_pos[obj_type]
        elif obj_type in op_types_pos:
            return op_types_pos[obj_type]
        elif obj_type in prop_types_pos:
            return prop_types_pos[obj_type]
        else:
            return []
