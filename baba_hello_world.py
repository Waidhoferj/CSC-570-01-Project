import random
import sys
sys.path.append("baba-is-auto/Extensions/BabaRL/baba-babaisyou-v0")
import environment

import rendering

import pyBaba
import numpy as np
import gym

env = gym.make('baba-babaisyou-v0')

if __name__ == '__main__':
    print("works")
    state = env.reset().reshape(1, -1, 9, 11)
    env.render()

    # global_step = 0

    # scores = []
    # for e in range(10000):
    #     score = 0

    #     state = env.reset().reshape(1, -1, 9, 11)
    #     state = torch.tensor(state).to(device)

    #     step = 0
    #     while step < 200:
    #         global_step += 1

    #         action = get_action(state)

    #         env.render()

    #         next_state, reward, done, _ = env.step(action)
    #         next_state = next_state.reshape(1, -1, 9, 11)
    #         next_state = torch.tensor(next_state).to(device)

    #         memory.push(state, action, next_state, reward)
    #         score += reward
    #         state = next_state

    #         step += 1

    #         train()

    #         if env.done:
    #             break

    #     writer.add_scalar('Reward', score, e)
    #     writer.add_scalar('Step', step, e)
    #     writer.add_scalar('Epsilon', EPSILON, e)

    #     scores.append(score)

    #     print(
    #         f'Episode {e}: score: {score:.3f} time_step: {global_step} step: {step} epsilon: {EPSILON}')

    #     if np.mean(scores[-min(50, len(scores)):]) > 180:
    #         print('Solved!')
    #         torch.save(net.state_dict(), 'dqn_agent.bin')
    #         break

    #     if e % TARGET_UPDATE == 0:
    #         target_net.load_state_dict(net.state_dict())

    #         EPSILON *= EPSILON_DECAY
    #         EPSILON = max(EPSILON, MIN_EPSILON)
