import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
import gymnasium as gym



class Policy(nn.Module):
    def __init__(
            self,
            state_space:int,
            action_space:int):
        super().__init__()
        self.fn1 = nn.Linear(state_space,128)
        self.fn2 = nn.Linear(128,action_space)



    def forward(self,state):
        out = self.fn2(F.relu(self.fn1(state)))
        out = F.softmax(out,dim=-1)
        return out
    

ActionProbVals = namedtuple("ActionProbVals",('log_prob','value'))

class Agent:
    def __init__(
            self,
            state_space:int,
            action_space:int,
            gemma=0.9):
        self.policy = Policy(state_space,action_space)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=1e-2)

        self.gemma =gemma
        self.rewards:list[float] = []
        self.action_prob_val_records:list[ActionProbVals] = []



    def select_one_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy(state)

        m = torch.distributions.Categorical(action_probs)
        action = m.sample()

        self.action_prob_val_records.append(ActionProbVals(m.log_prob(action),0))
        return action.item()
    

    def end_episode(self):
        return_rewards = []

        R = 0
        for r in self.rewards[::-1]:
            R = r +self.gemma*R
            return_rewards.insert(0,R)


        return_rewards = torch.tensor(return_rewards)
        return_rewards = (return_rewards-return_rewards.mean())/(return_rewards.std()+1e-6)

        loses = []
        for (log_prob,_),return_reward in zip(self.action_prob_val_records,return_rewards):
            loss = -log_prob*return_reward
            loses.append(loss)


        loss = torch.cat(loses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        del self.rewards[:]
        del self.action_prob_val_records[:]

def train():
    env = gym.make("CartPole-v1")
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = Agent(state_space,action_space,gemma=0.99)
    episodes =2000
    for i_episode in range(episodes):
        episode_reward = 0
        state,_ = env.reset()
        for _ in range(10_000):
            action = agent.select_one_action(state)
            state,reward,terminated,truncated,_ = env.step(action)
            agent.rewards.append(reward)
            episode_reward+=reward

            if terminated or truncated:
                break

        agent.end_episode()

        if (i_episode+1)%50 ==0:
            print(f'at episode {i_episode},rewards are {episode_reward}')

        if episode_reward >475:
            print(f'get episode rewards {episode_reward},now stop at episode {i_episode}')
            break


        



    

