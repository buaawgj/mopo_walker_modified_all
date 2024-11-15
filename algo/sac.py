import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class SACPolicy(nn.Module):
    def __init__(
        self, 
        actor, 
        critic1, 
        critic2,
        true_valnet,
        actor_optim, 
        critic1_optim, 
        critic2_optim,
        val_optim,
        action_space,
        dist, 
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        device="cpu"
    ):
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.true_valnet, self.true_valnet_old = true_valnet, deepcopy(true_valnet)
        self.true_valnet_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim
        self.val_optim = val_optim
        
        self.actor_lr = self.actor_optim.param_groups[0]['lr']
        self.critic1_lr = self.critic1_optim.param_groups[0]['lr']
        self.critic2_lr = self.critic2_optim.param_groups[0]['lr']
        self.val_lr = self.val_optim.param_groups[0]['lr']

        self.action_space = action_space
        self.dist = dist

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        
        self.__eps = np.finfo(np.float32).eps.item()

        self._device = device
    
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.true_valnet_old.parameters(), self.true_valnet.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def forward(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def sample_action(self, obs, deterministic=False):
        action, _ = self(obs, deterministic)
        return action.cpu().detach().numpy()

    def linear_decay(self, epoch_i, epoch_num):
        current_actor_lr = self.actor_lr * (1 - 0.7 * epoch_i / epoch_num)
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = current_actor_lr
            
        current_critic1_lr = self.critic1_lr * (1- 0.7 * epoch_i / epoch_num)
        for param_group in self.critic1_optim.param_groups:
            param_group['lr'] = current_critic1_lr
            
        current_critic2_lr = self.critic2_lr * (1- 0.7 * epoch_i / epoch_num)
        for param_group in self.critic2_optim.param_groups:
            param_group['lr'] = current_critic2_lr

        current_val_lr = self.val_lr * (1 - 0.7 * epoch_i / epoch_num)
        for param_group in self.val_optim.param_groups:
            param_group['lr'] = current_val_lr

    def learn(self, data, fake_data):
        obs, actions, next_obs, terminals, rewards = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"]
        fake_obs, fake_actions, next_fake_obs, fake_terminals, fake_rewards = fake_data["observations"], \
            fake_data["actions"], fake_data["next_observations"], fake_data["terminals"], fake_data["rewards"]
        
        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)
        
        fake_rewards = torch.as_tensor(fake_rewards).to(self._device)
        fake_terminals = torch.as_tensor(fake_terminals).to(self._device)
        
        # update value network
        # compute the true value
        true_value = self.true_valnet(obs).flatten()
        
        with torch.no_grad():
            true_next_value = self.true_valnet_old(next_obs)
            true_value_target = rewards.flatten() + self._gamma * true_next_value.flatten()
        
        # value_loss = torch.mean((true_value_target - true_value)**2)
        value_loss = torch.mean(F.relu(true_value_target.detach() - true_value, inplace=False)**2)
        self.val_optim.zero_grad()
        value_loss.backward()
        self.val_optim.step()
        
        # update critic
        # compute current critic values
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        q1f, q2f = self.critic1(fake_obs, fake_actions).flatten(), self.critic2(fake_obs, fake_actions).flatten()
        
        # compute target critic values
        with torch.no_grad():
            next_fake_actions, next_fake_log_probs = self(next_fake_obs)
            next_qf = torch.min(
                self.critic1_old(next_fake_obs, next_fake_actions), self.critic2_old(next_fake_obs, next_fake_actions)
            ) - self._alpha * next_fake_log_probs
            target_qf = fake_rewards.flatten() + self._gamma * (1 - fake_terminals.flatten()) * next_qf.flatten()
            
        with torch.no_grad():
            next_q = self.true_valnet(next_obs)
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
            
        critic1_loss = ((q1 - target_q).pow(2)).mean() + ((q1f - target_qf).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        
        critic2_loss = ((q2 - target_q).pow(2)).mean() + ((q2f - target_qf).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        obs = torch.as_tensor(obs).to(self._device)
        fake_obs = torch.as_tensor(fake_obs).to(self._device)
        obs = torch.cat([obs, fake_obs], dim=0).to(self._device)
        a, log_probs = self(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result
