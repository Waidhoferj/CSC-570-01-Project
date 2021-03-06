# https://github.com/higgsfield/Imagination-Augmented-Agents/blob/master/3.environment-model.ipynb
# https://github.com/Olloxan/Pytorch-A2C

import numpy as np
import time
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

train_data, _ = train_test_levels()
for i,level in enumerate(train_data):
        env_name = f"baba-babaisyou{i}-v0"
        register_baba_env(env_name, path=level, enable_render=False, env_class_str="PropertyBasedEnv")

logger = Logger()
timer = myTimer()


USE_CUDA = torch.cuda.is_available()
num_envs = 32

logger = Logger()

# env_name = "baba-babaisyou-v0"
env_path = os.path.join("baba-is-auto", "Resources", "Maps", "baba_is_you.txt")



def make_cuda(input):
    if USE_CUDA:
        return input.cuda()
    return input





def train(env_name:str,actor_critic, optimizer, rollout,
    gamma = 0.99,
    entropy_coef = 0.01,
    value_loss_coef = 0.5,
    max_grad_norm = 0.5,
    num_steps = 10,
    num_frames = int(1e6)):

    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env
        return _thunk

    envs = [make_env() for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state_shape = envs.observation_space.shape

    # a2c hyperparams:
    

    # Init a2c and rmsprop

    if USE_CUDA:
        rollout.cuda()

    all_rewards = []
    all_losses = []

    state = envs.reset()

    state = torch.FloatTensor(np.float32(state))

    rollout.states[0].copy_(state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards = torch.zeros(num_envs, 1)

    timer.update(time.time())

    for i_update in range(num_frames):

        for step in range(num_steps):

            action = actor_critic.act(make_cuda(state))

            next_state, reward, finished, _ = envs.step(
                action.squeeze(1).cpu().data.numpy()
            )

            reward = torch.FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            finished_masks = torch.FloatTensor(1 - np.array(finished)).unsqueeze(1)

            final_rewards *= finished_masks
            final_rewards += (1 - finished_masks) * episode_rewards

            episode_rewards *= finished_masks

            finished_masks = make_cuda(finished_masks)

            state = torch.FloatTensor(np.float32(next_state))
            rollout.insert(step, state, action.data, reward, finished_masks)

        _, next_value = actor_critic(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)
        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            rollout.states[:-1].view(-1, *state_shape), rollout.actions.view(-1, 1)
        )

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef

        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        optimizer.step()

        # print information every 100 epochs
        if i_update % 100 == 0:
            all_rewards.append(final_rewards.mean())
            all_losses.append(loss.item())
            print("epoch %s. reward: %s" % (i_update, np.mean(all_rewards[-10:])))
            print("loss %s" % all_losses[-1])
            print("---------------------------")

            timer.update(time.time())
            timediff = timer.getTimeDiff()
            total_time = timer.getTotalTime()
            loopstogo = (num_frames - i_update) / 100
            estimatedtimetogo = timer.getTimeToGo(loopstogo)
            logger.printDayFormat("runntime last epochs: ", timediff)
            logger.printDayFormat("total runtime: ", total_time)
            logger.printDayFormat("estimated time to run: ", estimatedtimetogo)
            print("######## AC_BABAISYOU ########")

        rollout.after_update()

        # snapshot of weights, data and optimzer every 1000 epochs
        if i_update % 1000 == 0 and i_update > 0:
            logger.log(all_rewards, "Data/", "all_rewards_BABAISYOU.txt")
            logger.log(all_losses, "Data/", "all_losses_BABAISYOU.txt")
            logger.log_state_dict(
                actor_critic.state_dict(), "Data/actor_critic_BABAISYOU"
            )
            logger.log_state_dict(
                optimizer.state_dict(), "Data/actor_critic_optimizer_BABAISYOU"
            )

    # final save
    logger.log(all_rewards, "Data/", "all_rewards_BABAISYOU.txt")
    logger.log(all_losses, "Data/", "all_losses_BABAISYOU.txt")
    logger.log_state_dict(actor_critic.state_dict(), "Data/actor_critic_BABAISYOU")
    logger.log_state_dict(
        optimizer.state_dict(), "Data/actor_critic_optimizer_BABAISYOU"
    )

if __name__ == "__main__":
    use_weights = False
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 10
    num_frames = 100
    num_envs = 32
    num_actions = 4
    observation_space = (18,30,30)
    actor_critic = ActorCritic(observation_space, num_actions)
    ac_weights = "Data/actor_critic_BABAISYOU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_weights and os.path.exists(ac_weights):
        actor_critic.load_state_dict(
        torch.load(ac_weights)
    )
    optimizer = optim.Adam(actor_critic.parameters())

    actor_critic = make_cuda(actor_critic)

    rollout = RolloutStorage(num_steps, num_envs, observation_space)

    for i,level in enumerate(train_data):
        print(f"Training on level {i+1} of {len(train_data)}")
        env_name = f"baba-babaisyou{i}-v0"
        train(env_name, actor_critic, optimizer, rollout,
        gamma =gamma,
    entropy_coef =entropy_coef,
    value_loss_coef =value_loss_coef,
    max_grad_norm =max_grad_norm,
    num_steps = num_steps,
    num_frames = num_frames)