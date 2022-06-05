""" Given a map of a level, calculate a difficulty heuristic """
import sys, os

from baba_blocks import BabaBlock

def calculate_congestion(input) -> int:
    """ Returns an integer value that represents how congested a level is """

    if len(input) == 0:
        print("Error: Input level is empty. ")
        return -1

    level_size = len(input) * len(input[0])
    non_empty_blocks = 0

    for row in input:
        for block in row:
            if block != BabaBlock.ICON_EMPTY.value:
                non_empty_blocks += 1

    return 100 * non_empty_blocks / level_size

def calculate_rule_variability(start_input, end_input) -> int:
    """ Returns a value that represents the number of rules that have to be changed in order to complete the level """
    
    if len(start_input) == 0 or len(end_input) == 0:
        print("Error: input level(s) is empty.")
        return -1
    
    rules_start = []
    rules_end = []

    for i, row in enumerate(start_input):
        for j, ele in enumerate(row):
            if ele == BabaBlock.IS.value: # might be a rule!
                if (j > 0 and j < len(end_input)-1) and start_input[i][j-1] < 66 and start_input[i][j+1] < 110 :
                    rules_start.append( ((start_input[i][j-1]), (start_input[i][j]), (start_input[i][j+1])) )
                elif (i > 0 and i < len(end_input)-1) and start_input[i-1][j] < 66 and start_input[i+1][j] < 110:
                    rules_start.append( ((start_input[i-1][j]), (start_input[i][j]), (start_input[i+1][j])) )
    
    for i, row in enumerate(end_input):
        for j, ele in enumerate(row):
            # looping through entire level
            if ele == BabaBlock.IS.value: # might be a rule!
                if (j != 0 and j != len(end_input)-1):
                    # print(f"{end_input[i][j-1]} IS {end_input[i][j+1]}")
                    if end_input[i][j-1] < 66 and end_input[i][j+1] < 110 :
                        rules_end.append( ((end_input[i][j-1]), (end_input[i][j]), (end_input[i][j+1])) )
                elif (i != 0 and i != len(end_input)-1):
                    if end_input[i-1][j] < 66 and end_input[i+1][j] < 110:
                        rules_end.append( ((end_input[i-1][j]), (end_input[i][j]), (end_input[i+1][j])) )
    
    # print(rules_start, rules_end)
    return (set(rules_start) | set(rules_end)) - (set(rules_start) & set(rules_end)) # find number of rules that are different between start and end

def calculate_difficulty(input) -> int:
    """ Returns a value that defines the difficulty of the game """

    return 0

def levelfile_to_array(filepath) -> list:

    levellist = []

    if os.path.exists(filepath) is False:
        print(f"Error: File {filepath} does not exist.")
        return []

    with open(filepath) as levelfile:
        for line in levelfile.readlines():
            # split every line into a new list -- result will be 2-dimensional
            levellist.append([ int(x) for x in line.split(' ') ]) # split every element in each row

    return levellist

if __name__ == "__main__":
    print("Running Program: 'level_difficulty_calc.py'\n--------------------")

    level : int = 223
    start_level_filename = f'levels/out/{level}.txt'
    end_level_filename = f'levels/out/{level}_end.txt'

    start_level_arr = levelfile_to_array(start_level_filename)[1:]
    end_level_arr = levelfile_to_array(end_level_filename)[1:]

    # print(start_level_arr, end_level_arr) # DEBUG: print out level arrays

    changed_rules = calculate_rule_variability(start_level_arr, end_level_arr)
    print(f"{len(changed_rules)} rules were changed: {changed_rules}")

    congestion = calculate_congestion(start_level_arr)
    print(f"{congestion} is the congestion level of the level.")

    # difficulty thresholds:
    # < 33 = EASY
    # < 66 = MEDIUM
    # ELSE = HARD
    difficulty_score = (len(changed_rules) * 10 + congestion)
    print(f"difficulty score is {difficulty_score}")

    sys.exit(0)