from argparse import ArgumentParser
from enum import Enum, auto
import shutil
from typing import List
from bidict import bidict
import os
import re
import json
import pyBaba

def get_baba_is_auto_game_enum_elements()-> List[str]:
    """ """
    enums_dir = ["baba-is-auto","Includes", "baba-is-auto", "Enums"]
    files = ["NounType.def","OpType.def", "PropertyType.def", "IconType.def"]
    elements = []
    for filename in files:
        path = os.path.join(*enums_dir, filename)
        with open(path, "r") as f:
            for line in f.readlines():
                el = re.search(r"X\((.*)\)", line).groups()[0]
                elements.append(el)
    return elements


def create_tile_enum():
    i = 0
    elements = {}
    def inc():
        nonlocal i
        v = i
        i +=1
        return v
    auto_enums = [getattr(pyBaba.ObjectType, k) for k in dir(pyBaba.ObjectType) if not k.startswith("_") and not k.islower()]
    elements.update({str(el).split(".")[1]: el.value for el in auto_enums})
    i = len(elements)
    elements.update({"FLOOR": inc(), "ICON_FLOOR": inc(), "GOOP": inc(), "ICON_GOOP": inc(), "BORDER": inc(), "KILL": inc()})
            
    return Enum('Tile', elements)

Tile = create_tile_enum()


LevelMap = List[List[Tile]]

baba_is_auto_map = bidict({
    Tile(i) : i for i in range(len(get_baba_is_auto_game_enum_elements()))
})

keke_map = bidict({
    Tile.BORDER : '_',
    Tile.ICON_EMPTY : '.',
    Tile.ICON_BABA : 'b',
    Tile.BABA : 'B',
    Tile.IS : '1',
    Tile.YOU : '2',
    Tile.WIN : '3',
    Tile.ICON_SKULL : 's',
    Tile.SKULL : 'S',
    Tile.ICON_FLAG : 'f',
    Tile.FLAG : 'F',
    Tile.ICON_FLOOR : 'o',
    Tile.FLOOR : 'O',
    Tile.GRASS : 'a',
    Tile.ICON_GRASS : 'A',
    Tile.KILL : '4',
    Tile.ICON_LAVA : 'l',
    Tile.LAVA : 'L',
    Tile.PUSH : '5',
    Tile.ICON_ROCK : 'r',
    Tile.ROCK : 'R',
    Tile.STOP : '6',
    Tile.ICON_WALL : 'w',
    Tile.WALL : 'W',
    Tile.MOVE : '7',
    Tile.HOT : '8',
    Tile.MELT : '9',
    Tile.ICON_KEKE : 'k',
    Tile.KEKE : 'K',
    Tile.ICON_GOOP : 'g',
    Tile.GOOP : 'G',
    Tile.SINK : '0',
    Tile.ICON_LOVE : 'v',
    Tile.LOVE : 'V',
})

def read_keke_solutions(filename: str) -> List[str]:
    """ Reads in keke format file <filename> and outputs a list of level solutions """
    solutions = []

    with open(filename, 'r') as infile:
        keke_levels = json.load(infile)["levels"]
        for keke_level in keke_levels:
            solutions.append(keke_level["solution"])

    return solutions

def read_keke(filename:str) -> List[LevelMap]:
    """ Reads in keke format file <filename> and output a list of LevelMaps """
    levels = []
    with open(filename, "r") as f:
        keke_levels = json.load(f)["levels"]
        for keke_level in keke_levels:
            level = []
            for line in keke_level["ascii"].split("\n"):
                level.append([])
                for c in line:
                    if c == " ": c = "." # dobule mapping on empties
                    level[-1].append(keke_map.inverse[c])
            levels.append(level)
            
    return levels
                    
def write_keke(filename:str, level:LevelMap, solutions: List[str] = []):
    """ Writes level <level> to keke format stored in file <filename>"""
    converted_level = "\n".join(["".join([keke_map[tile] for tile in row]) for row in level])
    output = {
        "levels": [{"id": "", "name": "", "author": "", "ascii": converted_level, "solution": ""}]
    }
    with open(filename, "w") as f:
       json.dump(output, f)
            

def read_baba_is_auto(filename:str) -> LevelMap:
    """ read baba auto file <filename> and output a LevelMap """
    with open(filename, "r") as f:
        level = []
        level = [[baba_is_auto_map.inverse[int(num)]for num in line.split(" ")] for line in f.readlines()[1:]]
    return level

def write_baba_is_auto(filename:str, levels: List[LevelMap], solutions: List[str] = []):
    path, filename = os.path.split(filename)
    file, ext =  os.path.splitext(filename)
    folder = os.path.join(path, file)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)

    def get_tile(tile: Tile):
        if tile == Tile.BORDER:
            tile = Tile.ICON_WALL
        return str(baba_is_auto_map.get(tile, baba_is_auto_map[Tile.ICON_EMPTY]))

    for i, level in enumerate(levels):
        lines = []
        height, width = len(level),len(level[0])
        lines.append(f"{width} {height}")
        converted_level = [" ".join(map(get_tile,row)) for row in level]
        lines.extend(converted_level)
        with open(os.path.join(path, file, f"{i}{ext}"), "w") as f:
            f.writelines(line + "\n" for line in lines)
    
    for i, solution in enumerate(solutions): # write solutions to i_sol.txt files
        with open(os.path.join(path, file, f"{i}_sol{ext}"), "w") as f:
            f.write(solution)
    
    return

converters = {
    "keke": (read_keke, write_keke),
    "baba-is-auto": (read_baba_is_auto, write_baba_is_auto),
}

def convert(src:str,dst:str,src_format:str,dst_format:str, include_solns:bool):
    """ convert from <src> file from <src_format> format to <dst_format> format and store in <dst> file """
    reader, _ = converters[src_format] # select the input format
    _, writer = converters[dst_format] # select the output format
    levels = reader(src)
    solutions = []
    if include_solns is True and reader is read_keke:
        solutions = read_keke_solutions(src)
    writer(dst, levels, solutions)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source", help="File with level information")
    parser.add_argument("dest", help="Output file location and name")
    parser.add_argument("--source-format", default="keke")
    parser.add_argument("--dest-format", default="baba-is-auto")
    parser.add_argument("--include-solutions", default=True)

    args = parser.parse_args()

    # Example: python map_converter.py keke_level.json levels/out.txt
    convert(args.source, args.dest, args.source_format, args.dest_format, args.include_solutions)


