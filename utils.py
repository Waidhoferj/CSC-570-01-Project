import pyBaba

move_map = {
    0: pyBaba.Direction.RIGHT,
    1: pyBaba.Direction.LEFT,
    2: pyBaba.Direction.UP,
    3: pyBaba.Direction.DOWN,
}


def is_breaking_rule(action, your_positions, rule_positions, rule_direction):
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


def is_breaking_st_is_you(env, action, your_positions) -> bool:
    """
    Check if the action breaks a rule
    """

    # rule: _ is YOU
    rule_positions, rule_direction = env.get_rule_w_property(pyBaba.ObjectType.YOU)

    if not rule_positions:
        return True
    else:
        return is_breaking_rule(action, your_positions, rule_positions, rule_direction)


def are_pos_adjacent(op_obj_pos, prop_obj_pos):
    """
    Returns is_adjacent, direction in which it's next to each other
    """
    op_obj_x, op_obj_y = op_obj_pos
    prop_obj_x, prop_obj_y = prop_obj_pos

    if op_obj_x != prop_obj_x and op_obj_y != prop_obj_y:
        return False, None

    # top to bottom
    if op_obj_x == prop_obj_x and (op_obj_y + 1 == prop_obj_y):
        return True, pyBaba.RuleDirection.VERTICAL

    # left to right
    if op_obj_y == prop_obj_y and (op_obj_x + 1 == prop_obj_x):
        return True, pyBaba.RuleDirection.HORIZONTAL

    return False, None


def get_direction_between_two_positions(pos_1, pos_2):
    pos_1_x, pos_1_y = pos_1
    pos_2_x, pos_2_y = pos_2

    if pos_1_x - 1 == pos_2_x and pos_1_y == pos_2_y:
        return pyBaba.Direction.LEFT
    elif pos_1_x + 1 == pos_2_x and pos_1_y == pos_2_y:
        return pyBaba.Direction.RIGHT
    elif pos_1_x == pos_2_x and pos_1_y - 1 == pos_2_y:
        return pyBaba.Direction.UP
    elif pos_1_x == pos_2_x and pos_1_y + 1 == pos_2_y:
        return pyBaba.Direction.DOWN

    return None


def check_boundaries(curr_pos, m_width, m_height):
    curr_pos_x, curr_pos_y = curr_pos
    if curr_pos_x < 0 or curr_pos_x >= m_width:
        return False
    if curr_pos_y < 0 or curr_pos_y >= m_height:
        return False
    return True


def check_obstacles(curr_pos, dir, env):
    return env.game.CanMove(curr_pos[0], curr_pos[1], dir)


def can_push(env, curr_pos, moves_to_make_list, m_width, m_height):
    """
    moves_to_make: nested list
    """

    can_move = True
    pos_to_add = []
    moves_to_add = []

    for moves_to_make in moves_to_make_list:
        can_move = True
        count_num_back_steps = 0
        for next_move in moves_to_make:
            if next_move == pyBaba.Direction.UP:
                next_pos = (curr_pos[0], curr_pos[1] - 1)
            elif next_move == pyBaba.Direction.DOWN:
                next_pos = (curr_pos[0], curr_pos[1] + 1)
            elif next_move == pyBaba.Direction.LEFT:
                next_pos = (curr_pos[0] - 1, curr_pos[1])
            else:
                next_pos = (curr_pos[0] + 1, curr_pos[1])

            if check_boundaries(next_pos, m_width, m_height) and check_obstacles(
                curr_pos, next_move, env
            ):
                count_num_back_steps += 1
                pos_to_add.append(next_pos)
                moves_to_add.append(next_move)
                curr_pos = next_pos
            else:
                can_move = False

        if can_move:
            break

        if not can_move:
            for _ in range(count_num_back_steps):
                pos_to_add.pop()
                moves_to_add.pop()

    return can_move, pos_to_add, moves_to_add


# TODO: rename this
def get_moves_to_make(prev_dir, curr_dir):
    if not curr_dir:
        return []

    if prev_dir == pyBaba.Direction.RIGHT:
        if curr_dir == pyBaba.Direction.UP:
            moves_to_make = [[pyBaba.Direction.DOWN, pyBaba.Direction.RIGHT]]
        elif curr_dir == pyBaba.Direction.DOWN:
            moves_to_make = [[pyBaba.Direction.UP, pyBaba.Direction.RIGHT]]
        elif curr_dir == pyBaba.Direction.LEFT:
            moves_to_make = [
                [
                    pyBaba.Direction.UP,
                    pyBaba.Direction.RIGHT,
                    pyBaba.Direction.RIGHT,
                    pyBaba.Direction.DOWN,
                ],
                [
                    pyBaba.Direction.DOWN,
                    pyBaba.Direction.RIGHT,
                    pyBaba.Direction.RIGHT,
                    pyBaba.Direction.UP,
                ],
            ]
        else:
            moves_to_make = []
    elif prev_dir == pyBaba.Direction.LEFT:
        if curr_dir == pyBaba.Direction.UP:
            moves_to_make = [[pyBaba.Direction.DOWN, pyBaba.Direction.LEFT]]
        elif curr_dir == pyBaba.Direction.DOWN:
            moves_to_make = [[pyBaba.Direction.UP, pyBaba.Direction.LEFT]]
        elif curr_dir == pyBaba.Direction.RIGHT:
            moves_to_make = [
                [
                    pyBaba.Direction.UP,
                    pyBaba.Direction.LEFT,
                    pyBaba.Direction.LEFT,
                    pyBaba.Direction.DOWN,
                ],
                [
                    pyBaba.Direction.DOWN,
                    pyBaba.Direction.LEFT,
                    pyBaba.Direction.LEFT,
                    pyBaba.Direction.UP,
                ],
            ]
        else:
            moves_to_make = []
    elif prev_dir == pyBaba.Direction.UP:
        if curr_dir == pyBaba.Direction.LEFT:
            moves_to_make = [[pyBaba.Direction.RIGHT, pyBaba.Direction.UP]]
        elif curr_dir == pyBaba.Direction.RIGHT:
            moves_to_make = [[pyBaba.Direction.LEFT, pyBaba.Direction.UP]]
        elif curr_dir == pyBaba.Direction.DOWN:
            moves_to_make = [
                [
                    pyBaba.Direction.RIGHT,
                    pyBaba.Direction.UP,
                    pyBaba.Direction.UP,
                    pyBaba.Direction.LEFT,
                ],
                [
                    pyBaba.Direction.LEFT,
                    pyBaba.Direction.UP,
                    pyBaba.Direction.UP,
                    pyBaba.Direction.RIGHT,
                ],
            ]
        else:
            moves_to_make = []
    else:
        if curr_dir == pyBaba.Direction.LEFT:
            moves_to_make = [[pyBaba.Direction.RIGHT, pyBaba.Direction.DOWN]]
        elif curr_dir == pyBaba.Direction.RIGHT:
            moves_to_make = [[pyBaba.Direction.LEFT, pyBaba.Direction.DOWN]]
        elif curr_dir == pyBaba.Direction.UP:
            moves_to_make = [
                [
                    pyBaba.Direction.RIGHT,
                    pyBaba.Direction.DOWN,
                    pyBaba.Direction.DOWN,
                    pyBaba.Direction.LEFT,
                ],
                [
                    pyBaba.Direction.LEFT,
                    pyBaba.Direction.DOWN,
                    pyBaba.Direction.DOWN,
                    pyBaba.Direction.RIGHT,
                ],
            ]
        else:
            moves_to_make = []
    return moves_to_make


def get_reverse_action(action: pyBaba.Direction):
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
