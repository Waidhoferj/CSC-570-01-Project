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
from utils import train_test_levels

logger = Logger()
timer = myTimer()


USE_CUDA = torch.cuda.is_available()
num_envs = 32

logger = Logger()


def make_cuda(input):
    if USE_CUDA:
        return input.cuda()
    return input


def run_I2A(actor_critic, env):
    actor_critic.eval()

    state = env.reset()

    done = False

    history = {"steps": 0, "won": False, "score": 0}
    steps = 0
    reward = None
    score = 0
    while not done:
        steps +=1
        current_state = torch.FloatTensor(state)

        action = actor_critic.act(make_cuda(current_state.unsqueeze(0)))

        next_state, reward, done, _ = env.step(action.data[0][0])
        score += reward

        state = next_state

    return {"steps": steps, "won": reward > 0, "score": score}


if __name__ == "__main__":
    _, test_data = train_test_levels()
    num_actions = 4
    state_shape =  (18,30,30)

    # Init a2c and rmsprop
    actor_critic = ActorCritic(state_shape, num_actions)
    actor_critic = make_cuda(actor_critic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pretrained I2A agent...")
    actor_critic.load_state_dict(
        torch.load("./agents/I2A/Data/actor_critic_BABAISYOU", map_location=device)
    )
    level_data = {}

    for i, test in enumerate(test_data):
        env_name = f"baba-test{i}-v0"
        
        register_baba_env(env_name, test, enable_render=True)
        env = gym.make(env_name)

        history = run_I2A(actor_critic, env)
        level_data[test] = history

    