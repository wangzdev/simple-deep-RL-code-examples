from collections import deque,namedtuple
import gymnasium as gym
import ale_py
from gymnasium.wrappers import FrameStackObservation,GrayscaleObservation,ResizeObservation 
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def make_env(render_mode=None):
    env = gym.make('SpaceInvadersNoFrameskip-v4',render_mode=render_mode)

    # original size [210,160,3]
    env = ResizeObservation(env,(84,84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env,stack_size=4)
    return env



class DQN(nn.Module):
    def __init__(self,
                 n_observation_space:int,
                 n_action_space:int
                 ):
        super().__init__()
        # input shape [4,84,84]
        self.cn1 = nn.Conv2d(n_observation_space,32,kernel_size=7,stride=3) # [32,25,25]
        self.cn2 = nn.Conv2d(32,64,kernel_size=3,stride=2) # [64,12,12]
        self.cn3 = nn.Conv2d(64,64,kernel_size=1,stride=1) # [64,12,12]

        self.fc1 = nn.Linear(int(64*12*12),4096)
        self.fc2 = nn.Linear(4096,n_action_space)


    def forward(self,state):
        out = F.relu(self.cn1(state))
        out = F.relu(self.cn2(out))
        out = F.relu(self.cn3(out))

        out = out.view(out.shape[0],-1) # flattern
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    


Record = namedtuple("Record",("state","action","rewards","next_state","done"))

class ReplayBuffer:
    def __init__(self,capacity:int=1000):
        self.buffer = deque([],maxlen=capacity)

    def push(self,state,action,rewards,next_state,done):
        self.buffer.append((state,action,rewards,next_state,done))

    def sample(self,batch_size:int):
        states,actions,rewards,next_states,dones = zip(*random.sample(self.buffer,batch_size))
        return states,actions,rewards,next_states,dones
    

    def __len__(self):
        return len(self.buffer)
    


class DDQNAgent:
    def __init__(
            self,
            state_space:int,
            action_space:int,
            buffer_capcity:int =100_000,
            batch_size:int =64,
            gamma=0.99,
            lr=1e-4,
            tau=0.95,
            device ='cpu'
            ):
        self.policy = DQN(state_space,action_space)
        self.policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capcity)
        self.gamma = gamma
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        self.target_network = DQN(state_space,action_space)
        self.target_network.to(device)     
        self.target_network.load_state_dict(self.policy.state_dict())
        self.target_network.eval()


    def select_action(self,state,episilon=0.1):
        action = None
        if random.random() >episilon:
            with torch.inference_mode():
                out = self.policy(state)
                action = torch.max(out,dim=-1)[1].item()
        else:
            action = random.randrange(0,self.action_space)

        return action
    


    def learn(self):
        if len(self.replay_buffer)<self.batch_size:
            return
        
        states,actions,rewards,next_states,dones = self.replay_buffer.sample(self.batch_size)
        states = torch.cat(states,dim=0)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.cat(next_states,dim=0)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        next_actions = self.policy(next_states).max(1)[1].unsqueeze(1)
        with torch.inference_mode():
            next_values = self.target_network(next_states).gather(1,next_actions)

        target_values = rewards+self.gamma*next_values*(1-dones)
        predict_values = self.policy(states).gather(1,actions)

        loss = F.smooth_l1_loss(predict_values,target_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for t_param,p_param in zip(self.target_network.parameters(),self.policy.parameters()):
        #    t_param.data.copy_(t_param.data*self.tau+(1-self.tau)*p_param.data)



def train(buffer_size=50_000,device='cpu'):
    env = make_env()
    state_shape = env.observation_space.shape
    print(f'state shape:{state_shape}')
    action_space = env.action_space.n
    agent = DDQNAgent(4,action_space,device=device,buffer_capcity=buffer_size)

    start_episilon = 1.0
    end_episilon = 0.1
    episilon = start_episilon
    episilon_decay = 0.9995
    print_every = 1
    global_step =0
    update_every_steps = 1000

    for i_episode in range(3000):
        episode_reward =0
        state,_ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)/255.0

        while True:
            global_step+=1
            action = agent.select_action(state,episilon)
            episilon = max(end_episilon,episilon_decay*episilon)

            next_state,reward,terminated,truncated,_ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)/255.0
            episode_reward+=reward
            if global_step%100==0:
                print(f'episode:{i_episode},at global step {global_step},reward:{reward},episode reward:{episode_reward},episilon:{episilon}')

            done = terminated or truncated

            agent.replay_buffer.push(state,action,reward,next_state,done)
            agent.learn()

            if global_step%update_every_steps ==0:
                agent.target_network.load_state_dict(agent.policy.state_dict())

            if global_step%100_000 ==0:
                torch.save(agent.policy.state_dict(),f'policy_step_{global_step}.pth')


            if done:
                break

            state = next_state


        if (i_episode+1)%print_every ==0:
            print(f'at episode:{i_episode},episode reward:{episode_reward}')

        if (i_episode+1)%1000 ==0:
            torch.save(agent.policy.state_dict(),f'policy_{i_episode+1}.pth')


        
            

    





        





    











