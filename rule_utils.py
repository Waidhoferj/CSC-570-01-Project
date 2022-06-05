import pyBaba
import os
from environment import register_baba_env
import gym
import pdb
import time
import itertools
from path_utils import place_agent_and_push, get_start_push_pos
from utils import (
    are_pos_adjacent,
    check_boundaries,
    check_objects,
    check_obstacles,
    is_empty,
    move_map,
    is_breaking_st_is_you,
)


def create_win_rule(env, new_rules, agent, enable_render=False):
    # form win rule
    # get_possible_win_rules
    win_rules = filter(lambda new_rule: new_rule[-1] == pyBaba.WIN, new_rules)
    can_create_win_rule = False

    for win_rule in win_rules:
        noun_obj, op_obj, prop_obj = win_rule

        all_noun_pos = env.get_obj_positions(noun_obj)
        all_op_pos = env.get_obj_positions(op_obj)
        all_prop_pos = env.get_obj_positions(prop_obj)

        # just get the first one # TODO: in the future, this may be changed to choose the closest ones to you or ones that are next to one another
        noun_pos = all_noun_pos[0]
        op_pos = all_op_pos[0]
        prop_pos = all_prop_pos[0]

        # op, prop, noun
        can_create_win_rule = create_win_rule_helper(
            env,
            agent,
            op_obj,
            op_pos,
            all_op_pos,
            prop_obj,
            prop_pos,
            all_prop_pos,
            noun_pos,
            all_noun_pos,
            enable_render=enable_render,
        )
        if can_create_win_rule:
            break

        # op, noun, prop
        can_create_win_rule = create_win_rule_helper(
            env,
            agent,
            op_obj,
            op_pos,
            all_op_pos,
            noun_obj,
            noun_pos,
            all_noun_pos,
            prop_pos,
            all_prop_pos,
            pivot_obj_type="noun",
            enable_render=enable_render,
        )
        if can_create_win_rule:
            break

    return can_create_win_rule


def create_win_rule_helper(
    env,
    agent,
    obj_1,
    obj_1_pos,
    obj_1_all,
    obj_2,
    obj_2_pos,
    obj_2_all,
    obj_3_pos,
    obj_3_all,
    pivot_obj_type="prop",
    enable_render=False,
):
    """
    obj_1 is the one on the left or the above obj_2
    """

    m_width = env.game.GetMap().GetWidth()
    m_height = env.game.GetMap().GetHeight()

    can_create_win_rule = False
    # check if op and prop are next to one another
    pairs = itertools.combinations([(obj_1, obj_1_pos), (obj_2, obj_2_pos)], 2)

    last_goal_pos = []
    for pair in pairs:
        obj_1, obj_2 = pair

        obj_1_type, obj_1_pos = obj_1
        obj_2_type, obj_2_pos = obj_2

        is_adjacent, dir = (
            are_pos_adjacent(obj_1_pos, obj_2_pos)
            if pivot_obj_type == "prop"
            else are_pos_adjacent(obj_2_pos, obj_1_pos)
        )

        if is_adjacent:
            if dir == pyBaba.RuleDirection.VERTICAL:
                last_goal_pos = (
                    [obj_1_pos[0], obj_1_pos[1] - 1]
                    if pivot_obj_type == "prop"
                    else [obj_1_pos[0], obj_1_pos[1] + 1]
                )
            else:
                last_goal_pos = (
                    [obj_1_pos[0] - 1, obj_1_pos[1]]
                    if pivot_obj_type == "prop"
                    else [obj_1_pos[0] + 1, obj_1_pos[1]]
                )
            break

    if not last_goal_pos:
        obj_2_pos_x, obj_2_pos_y = obj_2_pos

        # run bfs to get the possible location to form a rule!
        found = False
        moves = []
        visited = {obj_1_pos}
        obj_2_path_to_goal = [(obj_2_pos_x, obj_2_pos_y)]
        queue = [[obj_2_path_to_goal, moves]]
        other_obj_1_goal_pos = []
        is_start = True

        while not found and len(queue):
            obj_2_path_to_goal, moves = queue.pop(0)

            curr_pos = obj_2_path_to_goal[-1]

            curr_pos_x, curr_pos_y = curr_pos

            other_obj_1_goal_pos, other_obj_2_goal_pos = can_create_rules(
                env, curr_pos, pivot_obj_type
            )

            if other_obj_1_goal_pos and other_obj_2_goal_pos:
                found = True
                # don't move if you can create a rule here
                if len(obj_2_path_to_goal) != 1:
                    your_curr_positions = agent.get_your_positions(env)
                    if not place_agent_and_push(
                        env,
                        your_curr_positions,
                        found_path=(obj_2_path_to_goal, moves),
                        enable_render=enable_render,
                    ):
                        found = False

                if found:
                    if other_obj_1_goal_pos not in obj_1_all:
                        your_curr_positions = agent.get_your_positions(env)
                        if not place_agent_and_push(
                            env,
                            your_curr_positions,
                            obj_1_pos,
                            other_obj_1_goal_pos,
                            enable_render=enable_render,
                        ):
                            found = False

                    if found:
                        if other_obj_2_goal_pos not in obj_2_all:
                            your_curr_positions = agent.get_your_positions(env)
                            if not place_agent_and_push(
                                env,
                                your_curr_positions,
                                obj_3_pos,
                                other_obj_2_goal_pos,
                                enable_render=enable_render,
                            ):
                                found = False

            right = (curr_pos_x + 1, curr_pos_y)
            left = (curr_pos_x - 1, curr_pos_y)
            up = (curr_pos_x, curr_pos_y - 1)
            down = (curr_pos_x, curr_pos_y + 1)

            next_positions = [right, left, up, down]

            for i in range(len(next_positions)):
                next_pos = next_positions[i]
                move_dir = move_map[i]

                if next_pos not in visited:
                    if is_start:
                        start_pos = get_start_push_pos(curr_pos, move_dir, env)
                        if not is_empty(start_pos, env):
                            continue

                    # neighbors
                    if (
                        check_boundaries(next_pos, m_width, m_height)
                        and check_obstacles(curr_pos, move_dir, env)
                        and check_objects(next_pos, env)
                        and not is_breaking_st_is_you(env, move_dir, [curr_pos])
                    ):
                        visited.add(next_pos)
                        queue.append(
                            [obj_2_path_to_goal + [next_pos], moves + [move_dir]]
                        )
            if is_start:
                is_start = False

        if found:
            print("found")
            can_create_win_rule = True

    else:
        if last_goal_pos not in obj_3_all:
            your_curr_positions = agent.get_your_positions(env)
            if place_agent_and_push(
                env,
                your_curr_positions,
                obj_3_pos,
                last_goal_pos,
                enable_render=enable_render,
            ):
                print("found")
                can_create_win_rule = True

    return can_create_win_rule


def can_create_rules(env, obj_pos, obj_type="prop", op_type=pyBaba.IS):
    if obj_type == "prop":
        left_1 = False
        left_2 = False

        hor_op_goal_pos = []
        hor_noun_goal_pos = []

        op_found_horizontally = False
        op_found_vertically = False

        if obj_pos[0] - 2 >= 0:
            left_1 = (
                env.game.GetMap()
                .At(obj_pos[0] - 1, obj_pos[1])
                .HasType(pyBaba.ICON_EMPTY)
            )

            if not left_1 and env.game.GetMap().At(obj_pos[0] - 1, obj_pos[1]).HasType(
                op_type
            ):
                op_found_horizontally = True
                left_1 = True

            left_2 = (
                env.game.GetMap()
                .At(obj_pos[0] - 2, obj_pos[1])
                .HasType(pyBaba.ICON_EMPTY)
            )

        if left_1 and left_2:
            hor_op_goal_pos = (obj_pos[0] - 1, obj_pos[1])
            hor_noun_goal_pos = (obj_pos[0] - 2, obj_pos[1])

        top_1 = False
        top_2 = False
        vert_op_goal_pos = []
        vert_noun_goal_pos = []

        if obj_pos[1] - 2 >= 0:
            top_1 = (
                env.game.GetMap()
                .At(obj_pos[0], obj_pos[1] - 1)
                .HasType(pyBaba.ICON_EMPTY)
            )

            if not top_1 and env.game.GetMap().At(obj_pos[0], obj_pos[1] - 1).HasType(
                op_type
            ):
                op_found_vertically = True
                top_1 = True

            top_2 = (
                env.game.GetMap()
                .At(obj_pos[0], obj_pos[1] - 2)
                .HasType(pyBaba.ICON_EMPTY)
            )

            if top_1 and top_2:
                vert_op_goal_pos = (obj_pos[0], obj_pos[1] - 1)
                vert_noun_goal_pos = (obj_pos[0], obj_pos[1] - 2)

        if op_found_horizontally:
            return hor_op_goal_pos, hor_noun_goal_pos
        elif op_found_vertically:
            return vert_op_goal_pos, vert_noun_goal_pos

        return hor_op_goal_pos, hor_noun_goal_pos
    elif obj_type == "noun":
        right_1 = False
        right_2 = False

        hor_op_goal_pos = []
        hor_prop_goal_pos = []

        op_found_horizontally = False
        op_found_vertically = False

        if obj_pos[0] + 2 >= 0:
            right_1 = (
                env.game.GetMap()
                .At(obj_pos[0] + 1, obj_pos[1])
                .HasType(pyBaba.ICON_EMPTY)
            )

            if not right_1 and env.game.GetMap().At(obj_pos[0] - 1, obj_pos[1]).HasType(
                op_type
            ):
                op_found_horizontally = True
                right_1 = True

            right_2 = (
                env.game.GetMap()
                .At(obj_pos[0] + 2, obj_pos[1])
                .HasType(pyBaba.ICON_EMPTY)
            )

        if right_1 and right_2:
            hor_op_goal_pos = (obj_pos[0] + 1, obj_pos[1])
            hor_prop_goal_pos = (obj_pos[0] + 2, obj_pos[1])

        bottom_1 = False
        bottom_2 = False
        vert_op_goal_pos = []
        vert_prop_goal_pos = []

        if obj_pos[1] + 2 >= 0:
            bottom_1 = (
                env.game.GetMap()
                .At(obj_pos[0], obj_pos[1] + 1)
                .HasType(pyBaba.ICON_EMPTY)
            )

            if not bottom_1 and env.game.GetMap().At(
                obj_pos[0], obj_pos[1] + 1
            ).HasType(op_type):
                op_found_vertically = True
                bottom_1 = True

            bottom_2 = (
                env.game.GetMap()
                .At(obj_pos[0], obj_pos[1] + 2)
                .HasType(pyBaba.ICON_EMPTY)
            )

            if bottom_1 and bottom_2:
                vert_op_goal_pos = (obj_pos[0], obj_pos[1] + 1)
                vert_prop_goal_pos = (obj_pos[0], obj_pos[1] + 2)

        if op_found_horizontally:
            return hor_op_goal_pos, hor_prop_goal_pos
        elif op_found_vertically:
            return vert_op_goal_pos, vert_prop_goal_pos

        return hor_op_goal_pos, hor_prop_goal_pos

    else:
        print("invalid obj type!")
        exit(-1)


if __name__ == "__main__":
    env_name = "baba-volcano-v0"
    env_path = os.path.join("baba-is-auto", "Resources", "Maps", "volcano.txt")

    env_name = "baba-level-v0"
    env_path = os.path.join("levels", "out", "2.txt")

    register_baba_env(env_name, env_path, enable_render=False)

    from path_utils import get_path_and_moves, get_agent_push_path

    env = gym.make(env_name)
    env.reset()

    op_goal_pos, noun_goal_pos = can_create_rules(env, (5, 5))

    # print(op_goal_pos, noun_goal_pos)

    # path = []
    # moves = []
    # box_start = (2, 4)
    # box_goal = (4, 3)

    # path, moves = get_path_and_moves(env, box_start, box_goal)
    # print(path, moves)

    # agent_path, agent_moves = get_agent_push_path(env, box_start, box_goal)

    # print(agent_path)
    # print(agent_moves)

# for move in agent_moves:
#     env.step(move)
#     env.render()
#     time.sleep(0.2)
