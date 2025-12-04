import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
import gymnasium as gym


class ActorCritic(nn.Module):
    def __init__(self,state_space:int,action_space:int,middle_size = 128):
        super().__init__()

        self.actor = nn.Sequential(nn.Linear(state_space,middle_size),
                                   nn.ReLU(),
                                   nn.Linear(middle_size,middle_size),
                                   nn.ReLU(),
                                   nn.Linear(middle_size,action_space))
        
        self.critic = nn.Sequential(nn.Linear(state_space,middle_size),
                                    nn.ReLU(),
                                    nn.Linear(middle_size,middle_size),
                                    nn.ReLU(),
                                    nn.Linear(middle_size,1))
        
        self.apply(self._init_weights)
        

    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def act(self,state):
        out = self.actor(state)
        probs = F.softmax(out,dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.detach(), log_prob.detach()
    


    def evaluate(self,old_states:torch.Tensor,old_actions:torch.Tensor):
        out = self.actor(old_states)
        probs = F.softmax(out,dim=-1)
        m = torch.distributions.Categorical(probs)
        log_probs = m.log_prob(old_actions)
        state_values = self.critic(old_states)
        return log_probs,m.entropy(),state_values
    


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.states = []
        self.rewards = []
        self.is_terminates = []


    def clear(self):
        del self.actions[:]
        del self.log_probs[:]
        del self.states[:]
        del self.rewards[:]
        del self.is_terminates[:]


class PPO:
    DBG_COUNT =0
    def __init__(
            self,
            action_space:int,
            state_space:int,
            actor_lr:float = 1e-3,
            critic_lr:float=1e-2,
            advantage_eps = 0.1,
            gamma:float=0.99,
            gae_lambda:float=0.8,
            entropy_coef:float =0.01,
            K_rollouts = 8,
            device = 'mps'
            ):
        self.policy = ActorCritic(state_space,action_space)
        self.policy.to(device)
        self.optimzer = optim.Adam([{"params":self.policy.actor.parameters(),'lr':actor_lr},
                                    {"params":self.policy.critic.parameters(),'lr':critic_lr}])
        
        self.rollout_buff = RolloutBuffer()
        self.advantage_eps = advantage_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.K_rollouts = K_rollouts
        self.device =device

        self.old_policy = ActorCritic(state_space,action_space)
        self.old_policy.to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    
    def select_action(self,state):
        with torch.inference_mode():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action,log_probs = self.old_policy.act(state)


        self.rollout_buff.states.append(state)
        self.rollout_buff.actions.append(action)
        self.rollout_buff.log_probs.append(log_probs)
        return action.item()
    

    def update(self):
        old_states = torch.stack(self.rollout_buff.states).to(self.device).squeeze().detach()
        old_actions = torch.stack(self.rollout_buff.actions).to(self.device).squeeze().detach()
        old_log_probs = torch.stack(self.rollout_buff.log_probs).to(self.device).squeeze().detach()


        with torch.inference_mode():
            state_values = self.policy.critic(old_states).squeeze()
            rollout_steps = len(self.rollout_buff.states)
            last_advantage =0
            advantages = []
            for i in reversed(range(rollout_steps)):
                if self.rollout_buff.is_terminates[i]:
                    next_state_value =0
                    last_advantage =0 # reset last advantage
                else:
                    if i < rollout_steps-1:
                        next_state_value = state_values[i+1]
                    else:
                        next_state_value = state_values[i]


                td =self.rollout_buff.rewards[i]+self.gamma*next_state_value - state_values[i]
                advantage = td +self.gamma*self.gae_lambda*last_advantage
                advantages.insert(0,advantage)
                last_advantage = advantage

        
        advantages = torch.tensor(advantages,dtype=torch.float32).to(self.device)
     

        advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)

        returns = advantages+state_values

        for _ in range(self.K_rollouts):
            log_probs,entropis,current_state_values = self.policy.evaluate(old_states,old_actions)
            log_probs = log_probs.squeeze()
            entropis = entropis.squeeze()
            current_state_values = current_state_values.squeeze()

            ratio = torch.exp(log_probs-old_log_probs)

            # calculate surrogate advantages
            surr1 = ratio*advantages
            surr2 = torch.clamp(ratio,1-self.advantage_eps,1+self.advantage_eps)*advantages

            policy_loss = (-1*torch.min(surr1,surr2)).mean()
            critic_loss = F.smooth_l1_loss(current_state_values,returns)
            entropy_loss = -1*entropis.mean()
            loss = policy_loss+critic_loss+self.entropy_coef*entropy_loss

            self.optimzer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(),max_norm=1.0)
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(),max_norm=1.0)
            self.optimzer.step()



        self.old_policy.load_state_dict(self.policy.state_dict())
        self.rollout_buff.clear()



def train():
    env = gym.make('CartPole-v1')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n 
    print(f'state_space {state_space},action space {action_space}')

    agent = PPO(state_space=state_space,action_space=action_space,actor_lr=1e-3,critic_lr=5e-3,device='cpu')
    update_steps = 2000 # update agent every these steps
    episode_rewards = deque(maxlen=100)
    time_step =0
    for i_episode in range(40000):
        state,_  = env.reset()
        episode_reward =0
        for _ in range(2000):
            action = agent.select_action(state)
            state,reward,terminated,truncated,_ = env.step(action)
            agent.rollout_buff.rewards.append(reward)
            agent.rollout_buff.is_terminates.append(terminated)

            episode_reward+=reward
            time_step+=1
            if time_step%update_steps==0:
                agent.update()

            if terminated or truncated:
                break


        episode_rewards.append(episode_reward)
        if (i_episode+1)%400 ==0:
            print(f'at episode {i_episode},episode reward:{episode_reward},mean rewards:{np.mean(episode_rewards)}')

        if np.mean(episode_rewards)>475:
           print(f'at episode {i_episode},mean rewards greater than 475:{np.mean(episode_rewards)}')
           break

    env.close()







