import pdb
import numpy as np
import pyBaba
import time

from utils import (
    check_boundaries,
    check_objects,
    check_obstacles,
    can_push,
    get_moves_to_make,
    is_empty,
    move_map,
    is_breaking_st_is_you,
)


def place_agent_and_push(
    env,
    agent_curr_position,
    obj_start_pos=[],
    obj_goal_pos=[],
    found_path=None,
    enable_render=False,
):
    agent_push_path, agent_push_moves = get_agent_push_path(
        env, obj_start_pos, obj_goal_pos, found_path
    )

    # agent can't go there :(
    if not agent_push_path:
        return False

    agent_push_start_pos = agent_push_path[0]

    # Check if agent is in the spot to push
    if not (
        np.array(
            [
                (np.array(agent_push_start_pos) == pos).all()
                for pos in agent_curr_position
            ]
        )
    ).any():
        # TODO: for now, just select the first one, but in the future, this could be chosen more wisely
        your_pos = tuple(agent_curr_position[0])

        # move agent to the right location first
        agent_path, agent_moves = get_path_and_moves(
            env, your_pos, agent_push_start_pos, flag="agent"
        )

        if agent_moves:
            # move agent
            for agent_move in agent_moves:
                env.step(agent_move)
                if enable_render:
                    env.render()
                    time.sleep(0.2)

            # push prop
            for agent_push_move in agent_push_moves:
                env.step(agent_push_move)
                env.render()
                time.sleep(0.2)

            return True
        else:
            return False
    else:
        # push prop
        for agent_push_move in agent_push_moves:
            env.step(agent_push_move)
            env.render()
            time.sleep(0.2)
    # don't need to move
    return True


def get_path_and_moves(env, start_pos, goal_pos, flag="obj"):
    # bfs

    # path, moves
    path = [start_pos]
    moves = []
    queue = [[path, moves]]
    visited = {start_pos}
    m_width, m_height = env.game.GetMap().GetWidth(), env.game.GetMap().GetHeight()

    is_start = True
    while queue:
        path, moves = queue.pop(0)

        curr_pos = path[-1]

        if curr_pos == goal_pos:
            return path, moves

        right = (curr_pos[0] + 1, curr_pos[1])
        left = (curr_pos[0] - 1, curr_pos[1])
        up = (curr_pos[0], curr_pos[1] - 1)
        down = (curr_pos[0], curr_pos[1] + 1)

        possible_moves = [right, left, up, down]

        move_dir = None

        for i in range(len(possible_moves)):
            next_pos = possible_moves[i]
            move_dir = move_map[i]

            if next_pos not in visited:
                if flag == "obj" and is_start:
                    start_pos = get_start_push_pos(curr_pos, move_dir, env)

                    if not is_empty(start_pos, env):
                        continue

                if (
                    check_boundaries(next_pos, m_width, m_height)
                    and check_obstacles(curr_pos, move_dir, env)
                    and check_objects(next_pos, env)
                    and not is_breaking_st_is_you(env, move_dir, [curr_pos])
                ):
                    visited.add(next_pos)
                    queue.append([path + [next_pos], moves + [move_dir]])

        if is_start:
            is_start = False
    return [], []  # unreachable


def get_start_push_pos(curr_pos, move_dir, env):
    m_width = env.game.GetMap().GetWidth()
    m_height = env.game.GetMap().GetHeight()

    if move_dir == pyBaba.Direction.RIGHT:
        # left
        next_pos = (curr_pos[0] - 1, curr_pos[1])

    elif move_dir == pyBaba.Direction.LEFT:
        # right
        next_pos = (curr_pos[0] + 1, curr_pos[1])
    elif move_dir == pyBaba.Direction.UP:
        # down
        next_pos = (curr_pos[0], curr_pos[1] + 1)
    else:
        # up
        next_pos = (curr_pos[0], curr_pos[1] - 1)

    if check_boundaries(next_pos, m_width, m_height):
        return next_pos
    else:
        return ()


def add_agent_path(env, path, moves):
    m_width = env.game.GetMap().GetWidth()
    m_height = env.game.GetMap().GetHeight()

    new_path = path.copy()
    new_moves = moves.copy()

    pos_incr = 0
    moves_incr = 0

    for i in range(len(path) - 1):
        curr_pos = path[i]

        curr_move = moves[i]
        next_move = moves[i + 1] if i != len(path) - 2 else None

        extra_agent_moves = get_moves_to_make(curr_move, next_move)

        can_move, pos_to_add, moves_to_add = can_push(
            env, curr_pos, extra_agent_moves, m_width, m_height
        )

        if can_move:
            new_path = (
                new_path[: pos_incr + i + 1] + pos_to_add + new_path[pos_incr + i + 1 :]
            )
            new_moves = (
                new_moves[: moves_incr + i + 1]
                + moves_to_add
                + new_moves[moves_incr + i + 1 :]
            )

            pos_incr += len(pos_to_add)
            moves_incr += len(moves_to_add)
        else:
            return [], []
    return new_path[:-1], new_moves


def get_agent_start_pos(path, moves):
    obj_start_pos = path[0]
    obj_start_move = moves[0]

    if obj_start_move == pyBaba.Direction.RIGHT:
        return (obj_start_pos[0] - 1, obj_start_pos[1])
    elif obj_start_move == pyBaba.Direction.LEFT:
        return (obj_start_pos[0] + 1, obj_start_pos[1])
    elif obj_start_move == pyBaba.Direction.UP:
        return (obj_start_pos[0], obj_start_pos[1] + 1)
    else:
        return (obj_start_pos[0], obj_start_pos[1] - 1)


def get_agent_push_path(env, obj_start=[], obj_goal=[], found_path=None):
    """
    Path of an agent that pushes an obj
    """

    if not found_path:
        obj_path, obj_moves = get_path_and_moves(env, tuple(obj_start), tuple(obj_goal))

    else:
        obj_path, obj_moves = found_path

    if not obj_path:
        return [], []

    path, agent_moves = add_agent_path(env, obj_path, obj_moves)

    agent_start_pos = get_agent_start_pos(obj_path, obj_moves)

    agent_path = [agent_start_pos] + path

    return agent_path, agent_moves
