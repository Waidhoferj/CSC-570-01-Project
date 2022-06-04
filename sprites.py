from glob import glob
import pyBaba
import pygame
import os
BLOCK_SIZE = 48

appdir = os.path.abspath(os.path.dirname(__file__))

class SpriteLoader:
    def __init__(self):

        self.icon_images = {}
        for path in glob("sprites/icon/*.gif"):
            fname = os.path.split(path)[-1].split(".")[0]
            obj_type = getattr(pyBaba.ObjectType, f"ICON_{fname}")
            self.icon_images[obj_type] = pygame.transform.scale(pygame.image.load(
                path), (BLOCK_SIZE, BLOCK_SIZE))

        
        self.text_images = {}
        for path in glob("sprites/text/*.gif"):
            fname = os.path.split(path)[-1].split(".")[0]
            obj_type = getattr(pyBaba.ObjectType, fname)
            self.text_images[obj_type] = pygame.transform.scale(pygame.image.load(
                path), (BLOCK_SIZE, BLOCK_SIZE))