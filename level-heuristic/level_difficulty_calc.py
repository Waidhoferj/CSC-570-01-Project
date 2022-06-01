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
            if BabaBlock.EMPTY is False:
                non_empty_blocks += 1

    return 100 * non_empty_blocks / level_size

def calculate_rule_variability(input) -> int:
    """ Returns a value that represents the number of rules that can be changed """
    
    if len(input) == 0:
        print("Error: input level is empty.")
        return -1
    
    

    return 0

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
            levellist.append([ x for x in line.split(' ') ]) # split every element in each row

    return levellist

if __name__ == "__main__":
    print("Running Program: 'level_difficulty_calc.py'\n--------------------")

    

    sys.exit(0)