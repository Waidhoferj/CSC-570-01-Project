# https://github.com/higgsfield/Imagination-Augmented-Agents/blob/master/3.environment-model.ipynb
# https://github.com/Olloxan/Pytorch-A2C

import numpy as np
from time import sleep
import os
import gym

import torch
import torch.nn as nn
import torch.optim as optim

from common.multiprocessing_env import SubprocVecEnv
from common.actor_critic import ActorCritic, RolloutStorage
from common.logger import Logger
from common.myTimer import myTimer
from environment import register_baba_env

logger = Logger()
timer = myTimer()


USE_CUDA = torch.cuda.is_available()
num_envs = 32

logger = Logger()

env_name = "baba-babaisyou-v0"
env_path = os.path.join("baba-is-auto", "Resources", "Maps", "baba_is_you.txt")
register_baba_env(env_name, env_path, enable_render=True)


def make_cuda(input):
    if USE_CUDA:
        return input.cuda()
    return input


if __name__ == "__main__":  # important for windows systems if subprocesses are run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)

    state_shape = env.observation_space.shape
    num_actions = len(env.action_space)  # .n

    # Init a2c and rmsprop
    actor_critic = ActorCritic(state_shape, num_actions)
    actor_critic = make_cuda(actor_critic)

    print("Loading pretrained I2A agent...")
    actor_critic.load_state_dict(
        torch.load("./agents/I2A/Data/actor_critic_BABAISYOU", map_location=device)
    )
    actor_critic.eval()

    state = env.reset()

    done = False

    while not done:
        current_state = torch.FloatTensor(state)

        action = actor_critic.act(make_cuda(current_state.unsqueeze(0)))

        next_state, reward, done, _ = env.step(action.data[0][0])

        env.render()
        sleep(0.2)

        state = next_state
