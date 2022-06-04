import pyBaba
import os
from environment import register_baba_env
import gym
import pdb
import time
import itertools
from path_utils import place_agent_and_push
from utils import (
    are_pos_adjacent,
    check_boundaries,
    check_obstacles,
    move_map,
    is_breaking_st_is_you,
)


def create_win_rule(env, new_rules, agent, enable_render=False):
    # form win rule
    # get_possible_win_rules
    win_rules = filter(lambda new_rule: new_rule[-1] == pyBaba.WIN, new_rules)
    can_create_win_rule = False

    m_width = env.game.GetMap().GetWidth()
    m_height = env.game.GetMap().GetHeight()

    for win_rule in win_rules:
        noun_obj, op_obj, prop_obj = win_rule

        all_noun_pos = env.get_obj_positions(noun_obj)
        all_op_pos = env.get_obj_positions(op_obj)
        all_prop_pos = env.get_obj_positions(prop_obj)

        # just get the first one # TODO: in the future, this may be changed to choose the closest ones to you or ones that are next to one another
        noun_pos = all_noun_pos[0]
        op_pos = all_op_pos[0]
        prop_pos = all_prop_pos[0]

        print((op_obj, op_pos), (prop_obj, prop_pos))
        # check if op and prop are next to one another
        pairs = itertools.combinations([(op_obj, op_pos), (prop_obj, prop_pos)], 2)

        noun_goal_pos = []
        for pair in pairs:
            op_obj, prop_obj = pair

            op_obj_type, op_obj_pos = op_obj
            prop_obj_type, prop_obj_pos = prop_obj

            is_adjacent, dir = are_pos_adjacent(op_obj_pos, prop_obj_pos)

            if is_adjacent:
                if dir == pyBaba.RuleDirection.VERTICAL:
                    noun_goal_pos = [op_obj_pos[0], op_obj_pos[1] - 1]
                else:
                    noun_goal_pos = [op_obj_pos[0] - 1, op_obj_pos[1]]
                break

        # if op and prop are not next to each other
        if not noun_goal_pos:
            # check if prop can have two spaces to the left or to the top
            # if not move the prop
            prop_pos_x, prop_pos_y = prop_pos

            # run bfs to get the possible location to form a rule!
            found = False
            moves = []
            visited = {prop_pos}
            prop_path_to_goal = [(prop_pos_x, prop_pos_y)]
            queue = [[prop_path_to_goal, moves]]
            op_goal_pos = []

            while not found and len(queue):
                prop_path_to_goal, moves = queue.pop(0)

                curr_pos = prop_path_to_goal[-1]

                curr_pos_x, curr_pos_y = curr_pos

                op_goal_pos, noun_goal_pos = can_create_rules(env, curr_pos)

                if op_goal_pos and noun_goal_pos:
                    print("here")
                    print(op_goal_pos, noun_goal_pos, prop_path_to_goal)
                    found = True
                    # move prop
                    # don't move if you can create a rule here
                    if len(prop_path_to_goal) != 1:
                        your_curr_positions = agent.get_your_positions(env)
                        if not place_agent_and_push(
                            env,
                            your_curr_positions,
                            found_path=(prop_path_to_goal, moves),
                            enable_render=enable_render,
                        ):
                            found = False

                    if found:
                        if op_goal_pos not in all_op_pos:
                            your_curr_positions = agent.get_your_positions(env)
                            # move op
                            if not place_agent_and_push(
                                env,
                                your_curr_positions,
                                op_pos,
                                op_goal_pos,
                                enable_render=enable_render,
                            ):
                                found = False

                        if found:
                            if noun_goal_pos not in all_noun_pos:
                                # move noun
                                your_curr_positions = agent.get_your_positions(env)
                                if not place_agent_and_push(
                                    env,
                                    your_curr_positions,
                                    noun_pos,
                                    noun_goal_pos,
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
                        # neighbors
                        if (
                            check_boundaries(next_pos, m_width, m_height)
                            and check_obstacles(curr_pos, move_dir, env)
                            and not is_breaking_st_is_you(env, move_dir, [curr_pos])
                        ):
                            visited.add(next_pos)
                            queue.append(
                                [prop_path_to_goal + [next_pos], moves + [move_dir]]
                            )

            if found:
                print("found")
                can_create_win_rule = True
                break

        else:
            if noun_goal_pos not in all_noun_pos:
                # push noun goals
                # move noun
                your_curr_positions = agent.get_your_positions(env)
                if place_agent_and_push(
                    env,
                    your_curr_positions,
                    noun_pos,
                    noun_goal_pos,
                    enable_render=enable_render,
                ):
                    can_create_win_rule = True
                    break

    return can_create_win_rule


def can_create_rules(env, prop_pos):
    left_1 = False
    left_2 = False

    op_goal_pos = []
    noun_goal_pos = []
    if prop_pos[0] - 2 >= 0:
        left_1 = (
            env.game.GetMap()
            .At(prop_pos[0] - 1, prop_pos[1])
            .HasType(pyBaba.ICON_EMPTY)
        )
        # TODO: check if this space is reachable!
        left_2 = (
            env.game.GetMap()
            .At(prop_pos[0] - 2, prop_pos[1])
            .HasType(pyBaba.ICON_EMPTY)
        )

    if left_1 and left_2:
        op_goal_pos = (prop_pos[0] - 1, prop_pos[1])
        noun_goal_pos = (prop_pos[0] - 2, prop_pos[1])
    else:
        top_1 = False
        top_2 = False

        if prop_pos[1] - 2 >= 0:
            top_1 = (
                env.game.GetMap()
                .At(prop_pos[0], prop_pos[1] - 1)
                .HasType(pyBaba.ICON_EMPTY)
            )
            top_2 = (
                env.game.GetMap()
                .At(prop_pos[0], prop_pos[1] - 2)
                .HasType(pyBaba.ICON_EMPTY)
            )

        if top_1 and top_2:
            op_goal_pos = (prop_pos[0], prop_pos[1] - 1)
            noun_goal_pos = (prop_pos[0], prop_pos[1] - 2)

    return op_goal_pos, noun_goal_pos


if __name__ == "__main__":
    env_name = "baba-volcano-v0"
    env_path = os.path.join("baba-is-auto", "Resources", "Maps", "volcano.txt")

    register_baba_env(env_name, env_path, enable_render=False)

    from path_utils import get_path_and_moves, get_agent_push_path

    env = gym.make(env_name)
    env.reset()

    path = []
    moves = []
    box_start = (2, 4)
    box_goal = (4, 3)

    path, moves = get_path_and_moves(env, box_start, box_goal)
    print(path, moves)

    agent_path, agent_moves = get_agent_push_path(env, box_start, box_goal)

    print(agent_path)
    print(agent_moves)

# for move in agent_moves:
#     env.step(move)
#     env.render()
#     time.sleep(0.2)
