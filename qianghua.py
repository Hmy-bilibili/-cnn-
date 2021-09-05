import argparse
import gym
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from vec_env import VecFrameStack, SubprocVecEnv
import numpy as np
import gym
# from tensorboard_easy import Logger

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=520, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=False, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--num_env', type=int, default=8,
                    help='')
args = parser.parse_args(args=[])


class pWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(pWrapper, self).__init__(env)

    def observation(self, obs):
        obs = obs[35:195]
        obs = obs[::2, ::2, 0]
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        return obs.astype(np.float).ravel()


def make_envs(num_env, seed):
    def make_env(seed):
        def _thunk():
            env = gym.make('Pong-v0')
            env.seed(seed)
            env = pWrapper(env)
            return env

        return _thunk

    return SubprocVecEnv([make_env(seed + i) for i in range(num_env)])


nenv = args.num_env
# env = make_envs(nenv, args.seed)


def prepro(Is):
    """ prepro 210x160x3 into 6400 """
    I1 = np.array((32, 160, 160, 3))
    I2 = np.array((32, 80, 80, 1))
    I3 = np.array((32, 6400))
    for i in range(len(Is)):
        I1[i] = Is[i][35:195]
        I2[i] = I1[i][::2, ::2, 0]
        I2[i][I2[i] == 144] = 0
        I2[i][I2[i] == 109] = 0
        I2[i][I2[i] != 0] = 1
        I3[i] = I2[i].astype(np.float).ravel()
    return I3


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(200, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))

    return [action[0][i].item() + 1 for i in range(nenv)]


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob.double() * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    for i_episode in count(1):
        print("Episode ", i_episode)
        state, ep_reward = env.reset(), np.array([0 for i in range(nenv)], dtype=np.float64)
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += np.array(reward)
            if all(done):
                env.reset()
                break

        finish_episode()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward.mean()))
        mean_reward = ep_reward.mean()
        torch.save(policy.state_dict(), 'params.pth')

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # with Logger('./logs/') as log:
        #     log.log_scalar('Mean Reward', mean_reward, step=i_episode)


if __name__ == '__main__':
    env = make_envs(nenv, args.seed)
    main()