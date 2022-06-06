import pdb
import torch
from torch import nn, optim
import torch.nn.functional as F

from collections import namedtuple
import random
import numpy as np

import gym
import environment
import pyBaba
import os
import time

from time import sleep
from typing import List

# from tensorboardX import SummaryWriter
from environment import register_baba_env
from utils import train_test_levels


class IDAQStarAgent:
    def __init__(self, q_func):
        self.optimal_moves = []  # the final set of most optimal moves
        self.q_func = q_func

    # combine different heuristics such as being me a goal or reaching a goal
    def heuristic(self, state: np.array) -> torch.Tensor:
        return self.q_func(state)

    def get_env_game_state(self, env):
        game_play_state = env.game.GetPlayState()
        game_objects = env.game.GetMap().GetObjects()
        game_rules = env.game.GetRuleManager().GetPropertyRules()
        game_player_icon = env.game.GetPlayerIcon()

        return (game_play_state, game_objects, game_rules, game_player_icon)

    def simulate(self, env: gym.Env) -> bool:
        """
        Makes a move in the environment
        Args:
            env: The environment where the agent will take an action
        Returns:
            Whether the environment is at a final state
        """

        state = env.get_obs()
        threshold = torch.min(self.heuristic(state))

        moves_taken = 0
        optimal_moves = []
        i = 0

        while True:
            possible_env = gym.make(env_name)
            possible_env.reset()

            visited = {tuple(possible_env.get_obs().reshape(-1).tolist())}

            temp = self.search(
                possible_env, moves_taken, threshold, optimal_moves, visited
            )

            # when the goal is found
            if temp == -1:
                self.optimal_moves = optimal_moves
                print(self.optimal_moves)
                return

            # can't find the optimal moves
            if temp == np.inf:
                return

            threshold = temp

    def can_move(self, env, action) -> bool:
        """
        Check if the action breaks a rule
        """

        # cost of breaking a property rule
        positions = self.get_your_positions(env)

        # rule: _ is YOU
        rule_positions, rule_direction = self.get_st_is_you_positions(env)

        if not rule_positions:
            return True
        else:
            return not self.is_breaking_rule(
                action, positions, rule_positions, rule_direction
            )

    def is_breaking_rule(self, action, your_positions, rule_positions, rule_direction):
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

    def get_st_is_you_positions(self, env):
        rule_st_is_you = env.game.GetRuleManager().GetRules(pyBaba.ObjectType.YOU)[
            0
        ]  # returns [BABA, IS, YOU]
        rule_objs = rule_st_is_you.GetObjects()

        rule_positions_cand = [
            env.game.GetMap().GetPositions(rule_obj.GetTypes()[0])
            for rule_obj in rule_objs
        ]

        if len(rule_positions_cand[-1]) > 1:
            return None, None

        # check left to right
        you_pos = rule_positions_cand[-1][0]
        is_pos = (you_pos[0] - 1, you_pos[1])

        if is_pos in rule_positions_cand[1]:
            obj_pos = (is_pos[0] - 1, is_pos[1])
            if obj_pos in rule_positions_cand[0]:
                return [obj_pos, is_pos, you_pos], pyBaba.RuleDirection.HORIZONTAL

        # check top to bottom
        is_pos = (you_pos[0], you_pos[1] - 1)

        if is_pos in rule_positions_cand[1]:
            obj_pos = (is_pos[0], is_pos[1] - 1)
            if obj_pos in rule_positions_cand[0]:
                return [obj_pos, is_pos, you_pos], pyBaba.RuleDirection.VERTICAL

        return None, None

    def search(self, env, g_score, threshold, optimal_moves, visited):
        # if the current position is goal
        if env.game.GetPlayState() == pyBaba.PlayState.WON:
            return -1

        min_cost = np.inf

        for idx in len(env.action_space):

            action = env.action_space[idx]

            if self.can_move(env, action):
                prev = env.get_obs()
                curr_obs, _, done, _ = env.env.step(action)

                q_values = self.heuristic(curr_obs)

                predicted_cost = g_score + q_values[idx]

                if predicted_cost > threshold:
                    return predicted_cost

                # check in case the agent didn't move because it's blocked
                is_moved = (prev != curr_obs).any()

                if done:

                    playState = env.game.GetPlayState()

                    # win
                    if playState == pyBaba.PlayState.WON:
                        optimal_moves.append(action)
                        return -1

                    # when the timelimit is signalled, but game is still playing
                    elif playState == pyBaba.PlayState.PLAYING:
                        return min_cost

                    # lose
                    else:
                        return np.inf

                # game_state = self.get_env_game_state(env)
                state = tuple(curr_obs.reshape(-1).tolist())

                if state not in visited:
                    visited.add(state)
                    optimal_moves.append(action)

                    env.render()

                    temp = self.search(
                        env, g_score + 1, threshold, optimal_moves, visited
                    )

                    if temp == -1:
                        return temp

                    if temp < min_cost:
                        min_cost = temp

                    optimal_moves.pop()

                if is_moved:
                    # backtrack

                    back_action = self.get_back_action(action)
                    env.step(back_action)

        return min_cost

    def get_back_action(self, action: pyBaba.Direction):
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

    def step(self, env: gym.Env):
        action = self.optimal_moves.pop(0)
        print(action)
        _, reward, done, _ = env.step(action)
        return reward, done

    def get_your_positions(self, env: gym.Env) -> List[np.array]:
        positions = env.game.GetMap().GetPositions(env.game.GetPlayerIcon())
        positions = [np.array(p) for p in positions]
        return positions

    def get_goal_positions(self, env: gym.Env) -> np.array:
        rule_manager = env.game.GetRuleManager()
        win_rules = rule_manager.GetRules(pyBaba.ObjectType.WIN)

        convert = pyBaba.ConvertTextToIcon
        win_positions = []
        game_map = env.game.GetMap()

        for win_rule in win_rules:

            win_rule_type = win_rule.GetObjects()[0].GetTypes()[0]

            win_positions.extend(game_map.GetPositions(convert(win_rule_type)))

        return np.array(win_positions)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(
            18, 64, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, 1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(1)

        self.fc = nn.Linear(900, 4)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.data.size(0), -1)
        return self.fc(x)


class DQN:
    def __init__(self, net, target_net):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPSILON = 0.9
        self.EPSILON_DECAY = 0.99
        self.MIN_EPSILON = 0.01
        self.TARGET_UPDATE = 10

        self.net = net
        self.target_net = target_net

        self.target_net.load_state_dict(net.state_dict())
        self.target_net.eval()

        self.opt = optim.Adam(net.parameters())
        self.memory = ReplayMemory(10000)

    def get_action(self, state):
        if random.random() > self.EPSILON:
            with torch.no_grad():
                return env.action_space[self.net(state).max(1)[1].view(1)]
        else:
            return random.choice(env.action_space)

    def train(self, env):
        # writer = SummaryWriter()

        global_step = 0

        scores = []
        for e in range(10000):
            score = 0

            state = np.expand_dims(env.reset(),axis=0)
            state = torch.tensor(state).to(device)

            step = 0
            while step < 200:
                global_step += 1

                action = self.get_action(state)

                # env.render()

                next_state, reward, done, _ = env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                next_state = torch.tensor(next_state).to(device)

                self.memory.push(state, action, next_state, reward)
                score += reward
                state = next_state

                step += 1

                self.__train()

                if env.done:
                    break

            # writer.add_scalar("Reward", score, e)
            # writer.add_scalar("Step", step, e)
            # writer.add_scalar("Epsilon", EPSILON, e)

            scores.append(score)

            print(
                f"Episode {e}: score: {score:.3f} time_step: {global_step} step: {step} epsilon: {self.EPSILON}"
            )

            if np.mean(scores[-min(50, len(scores)) :]) > 180:
                print("Solved!")
                torch.save(self.net.state_dict(), "dqn_agent.bin")
                break

            if e % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.net.state_dict())

                self.EPSILON *= self.EPSILON_DECAY
                self.EPSILON = max(self.EPSILON, self.MIN_EPSILON)

    def __train(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[int(a) - 1]]), batch.action)))
        rewards = tuple(
            (map(lambda r: torch.tensor([r], dtype=torch.float32), batch.reward))
        )

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions).to(device)
        reward_batch = torch.cat(rewards).to(device)

        q_values = self.net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_q_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        expected_state_action_values = (next_q_values * self.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(q_values, expected_state_action_values.unsqueeze(1))

        self.opt.zero_grad()
        loss.backward()

        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.opt.step()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
    train, test = train_test_levels()
    # Train DQN
    net = Network().to(device)
    target_net = Network().to(device)
    dqn = DQN(net, target_net)
    should_train = True
    pretrained_model_path = "weights/dqn_agent.pth"
    if not should_train and  os.path.exists(pretrained_model_path):
        net.load_state_dict(pretrained_model_path)
        net.eval()
    if should_train:
        print("Training")
        for i,level in enumerate(train):
            print(f"Training on level {i}")
            score_history = []
            env_name = f"baba-train{i}-v0"
            env_template = register_baba_env(env_name, path=level, enable_render=False, env_class_str="PropertyBasedEnv")
            env = gym.make(env_name)
            for i in range(5):
                dqn.train(env)
            torch.save(net.state_dict(), pretrained_model_path)

    # Used trained DQN as a heuristic for IDA*
    done = False
    level_performance = {}
    agent = IDAQStarAgent(q_func=net)

    print("TEST")
    for i,level in enumerate(test):
        env_name = f"test-test{i}-v0"
        env_template = register_baba_env(env_name, path=level, enable_render=True, env_class_str="PropertyBasedEnv")
        env = gym.make(env_name)
        observation = (env.reset()).flatten()
        reward = 0
        score = 0
        steps = 0
        done = False
        agent.simulate(env)
        while not done:
            reward, done = agent.step(env)
            score += reward
            steps +=1
        level_performance[level] = {"score": score, "steps": steps, "won": reward > 0}

        print(level_performance[level])

    start_time = time.time()
    
    print(f"Total simulation time: {time.time() - start_time}s")

    while not done:
        
        env.render()
        sleep(0.2)
