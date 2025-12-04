import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple,deque
import gymnasium as gym



class ActorCritic(nn.Module):
    def __init__(
            self,
            state_space:int,
            action_space:int,
            middle_sz:int = 128
            ):
        super().__init__()
        # a shared layer to learn common things
        self.shared_layer1 = nn.Linear(state_space,middle_sz)
        self.actor = nn.Linear(middle_sz,action_space)
        self.critic = nn.Linear(middle_sz,1)


    def forward(self,state):
        out = F.relu(self.shared_layer1(state))
        action = self.actor(out)
        value = self.critic(out)

        action_probs = F.softmax(action,dim=-1)
        return action_probs,value
    



class A2CAgent:
    PRINT_COUNT = 0
    def __init__(
            self,
            state_space:int,
            action_space:int,
            gemma = 0.99,
            n_steps:int =16,
            lr =1e-4,
            entropy_coef = 0.01
            ):
        self.agent = ActorCritic(state_space,action_space)
        self.optimizer = optim.AdamW(self.agent.parameters(),lr=lr)

        self.memory= []
        self.gemma = gemma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef


    def select_action(self,state):
        state = torch.tensor(state).float().unsqueeze(0)

        action_probs,value = self.agent(state)

        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        action_log_prob = m.log_prob(action)
        return action.item(),action_log_prob,value.squeeze(),m.entropy()
    

    def learn(self,last_state_value):
        returns = deque()
        R = last_state_value.item()
        for _,_,_,_,reward in self.memory[::-1]:
            R = reward+self.gemma*R
            returns.appendleft(R)

        returns = torch.tensor(list(returns))

        log_probs = torch.stack([mem[1] for mem in self.memory]).squeeze()
        values = torch.stack([mem[2] for mem in self.memory])
        entropies = torch.stack([mem[3] for mem in self.memory])

        if A2CAgent.PRINT_COUNT<5:
            A2CAgent.PRINT_COUNT+=1
            print(f'log_probs shape:{log_probs.shape},values shape:{values.shape},entropies shape:{entropies.shape}')

        
        advantages = returns -values
        if len(self.memory) >1:
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)

        policy_loss = (-1*log_probs*advantages.detach()).mean()
        value_loss = F.smooth_l1_loss(values,returns)
        entropy_loss = -entropies.mean()

        loss = policy_loss +value_loss +self.entropy_coef*entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(),max_norm=1.0)
        self.optimizer.step()

        del self.memory[:]



def train():
    env = gym.make('CartPole-v1')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    a2c_agent = A2CAgent(state_space,action_space,lr=1e-3)

    scores = deque(maxlen=100) # only keep the latest 100 scores

    for i_episode in range(12000):
        state,_ = env.reset()
        episode_reward =0
        while True:
            for _ in range(a2c_agent.n_steps):
                action,action_log_prob,state_value,entropy = a2c_agent.select_action(state)
                next_state,reward,terminated,truncated,_ = env.step(action)
                episode_reward+=reward

                a2c_agent.memory.append((action,action_log_prob,state_value,entropy,reward))
                state = next_state
                if terminated or truncated:
                    break

            last_state_value = torch.tensor([0.0])
            if not (terminated or truncated):
                 _,_,last_state_value,_ = a2c_agent.select_action(state)

            a2c_agent.learn(last_state_value.detach())
            if terminated or truncated:
              break

        scores.append(episode_reward)

        if (i_episode+1)%100==0:
          print(f'at episode {i_episode},mean scores {np.mean(scores):.2f}')

        if np.mean(scores) >475:
          print(f'at episode {i_episode},mean scores pass 475 :{np.mean(scores)}')
          break

    env.close()



      















        






