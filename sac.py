import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ===================== Actor =====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob

    def act(self, state, deterministic=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean)  
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            return y_t

# ===================== Critic =====================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.q(x)

# ===================== Alpha =====================
class Alpha(nn.Module):
    def __init__(self, init_log_alpha=-1.5):
        super(Alpha, self).__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.float32))

    def forward(self):
        return self.log_alpha.exp()

# ===================== Replay Buffer =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        return (torch.tensor(state),
                torch.tensor(action),
                torch.tensor(reward).unsqueeze(1),
                torch.tensor(next_state),
                torch.tensor(done).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# ===================== SAC Optimization Step =====================
def optimize_sac(actor, critic1, critic2, target_critic1, target_critic2,
                 actor_optimizer, critic1_optimizer, critic2_optimizer,
                 alpha_module, alpha_optimizer,
                 replay_buffer, batch_size, gamma=0.99, tau=0.005,
                 target_entropy=-1.0, device='cpu', l2_reg=1e-4):  
    
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_states)
        next_q1 = target_critic1(next_states, next_actions)
        next_q2 = target_critic2(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2) - alpha_module() * next_log_probs
        target_q = rewards + gamma * (1 - dones) * next_q

    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)

    # ✅ L2 regularization terms
    l2_loss_critic1 = sum(param.pow(2).sum() for param in critic1.parameters() if param.ndim > 1)
    l2_loss_critic2 = sum(param.pow(2).sum() for param in critic2.parameters() if param.ndim > 1)

    critic1_loss = F.mse_loss(current_q1, target_q) + l2_reg * l2_loss_critic1
    critic2_loss = F.mse_loss(current_q2, target_q) + l2_reg * l2_loss_critic2

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    new_actions, log_probs = actor.sample(states)
    q1 = critic1(states, new_actions)
    q2 = critic2(states, new_actions)
    min_q = torch.min(q1, q2)
    actor_loss = (alpha_module() * log_probs - min_q).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    alpha_loss = -(alpha_module.log_alpha * (log_probs + target_entropy).detach()).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    with torch.no_grad():
        for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
            target_param.data.mul_(1 - tau).add_(tau * param.data)
        for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
            target_param.data.mul_(1 - tau).add_(tau * param.data)

    return (
        actor_loss.item(),
        critic1_loss.item(),
        critic2_loss.item(),
        alpha_module().item(),
        min_q.mean().item(),
        log_probs.mean().item(),
    )
