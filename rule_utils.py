import pyBaba
import itertools


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


def get_new_rules(noun_types, op_types, prop_types, curr_rules):
    cand_rules = []
    for comb in itertools.product(noun_types, op_types, prop_types):
        if comb not in curr_rules:
            cand_rules.append(comb)

    return cand_rules
