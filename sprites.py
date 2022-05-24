import pyBaba
import pygame
import os
BLOCK_SIZE = 48

appdir = os.path.abspath(os.path.dirname(__file__))

class SpriteLoader:
    def __init__(self):
        self.icon_images = {pyBaba.ObjectType.ICON_BABA: 'BABA',
                            pyBaba.ObjectType.ICON_FLAG: 'FLAG',
                            pyBaba.ObjectType.ICON_CRAB: 'CRAB',
                            pyBaba.ObjectType.ICON_ANNI: 'ANNI',
                            pyBaba.ObjectType.ICON_WATER: 'WATER',
                            pyBaba.ObjectType.ICON_FIRE: 'FIRE',
                            pyBaba.ObjectType.ICON_DUST: 'DUST',
                            pyBaba.ObjectType.ICON_WALL: 'WALL',
                            pyBaba.ObjectType.ICON_ROCK: 'ROCK',
                            pyBaba.ObjectType.ICON_TILE: 'TILE',
                            pyBaba.ObjectType.ICON_TYPE: 'TYPE_replacement',
                            pyBaba.ObjectType.ICON_LAVA: 'LAVA'}

        for i in self.icon_images:
            p, _ = os.path.split(__file__)
            fp = f'{p}/sprites/icon/{self.icon_images[i]}.gif'
            self.icon_images[i] = pygame.transform.scale(pygame.image.load(
                fp), (BLOCK_SIZE, BLOCK_SIZE))

        self.text_images = {pyBaba.ObjectType.BABA: 'BABA',
                            pyBaba.ObjectType.IS: 'IS',
                            pyBaba.ObjectType.YOU: 'YOU',
                            pyBaba.ObjectType.FLAG: 'FLAG',
                            pyBaba.ObjectType.CRAB: 'CRAB',
                            pyBaba.ObjectType.FIRE: 'FIRE',
                            pyBaba.ObjectType.DUST: 'DUST',
                            pyBaba.ObjectType.WIN: 'WIN',
                            pyBaba.ObjectType.WALL: 'WALL',
                            pyBaba.ObjectType.FACING: 'FACING',
                            pyBaba.ObjectType.STOP: 'STOP',
                            pyBaba.ObjectType.ROCK: 'ROCK',
                            pyBaba.ObjectType.PUSH: 'PUSH',
                            pyBaba.ObjectType.LAVA: 'LAVA',
                            pyBaba.ObjectType.MELT: 'MELT',
                            pyBaba.ObjectType.ANNI: 'ANNI',
                            pyBaba.ObjectType.WATER: 'WATER',
                            pyBaba.ObjectType.MOVE: 'MOVE',
                            pyBaba.ObjectType.HOT: 'HOT'}

        for i in self.text_images:
            p, _ = os.path.split(__file__)
            fp = f'{p}/sprites/text/{self.text_images[i]}.gif'
            self.text_images[i] = pygame.transform.scale(pygame.image.load(fp), (BLOCK_SIZE, BLOCK_SIZE))
