import numpy as np
import torch
import os
from common import util, functional
from models.ensemble_dynamics import EnsembleModel
from operator import itemgetter
from common.normalizer import StandardNormalizer
from copy import deepcopy
import torch.nn.functional as F


class TransitionModel:
    def __init__(self,
                 obs_space,
                 action_space,
                 static_fns,
                 true_valnet,
                 model_valnet,
                 lr,
                 holdout_ratio=0.1,
                 inc_var_loss=False,
                 use_weight_decay=False,
                 gamma = 0.99,
                 tau = 0.003,
                 **kwargs):

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.device = util.device
        self.model = EnsembleModel(obs_dim=obs_dim, action_dim=action_dim, device=util.device, **kwargs['model'])
        self.true_valnet, self.true_valnet_old = true_valnet, deepcopy(true_valnet)
        self.model_valnets = [deepcopy(model_valnet) for _ in range(self.model.ensemble_num)]
        self.model_valnet_olds = [deepcopy(model_valnet) for _ in range(self.model.ensemble_num)]
        self.static_fns = static_fns
        self.lr = lr
        self.gamma = gamma
        self._tau = tau

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.true_val_optimizer = torch.optim.Adam(self.true_valnet.parameters(), 0.5e-5)

        for i, model_valnet in enumerate(self.model_valnets):
            if i == 0:
                model_valnet_params = list(model_valnet.parameters())
            else:
                model_valnet_params += list(model_valnet.parameters())
        self.model_val_optimzer = torch.optim.Adam(model_valnet_params, 0.5e-5)
        
        self.networks = {
            "model": self.model
        }
        self.obs_space = obs_space
        self.holdout_ratio = holdout_ratio
        self.inc_var_loss = inc_var_loss
        self.use_weight_decay = use_weight_decay
        self.obs_normalizer = StandardNormalizer()
        self.act_normalizer = StandardNormalizer()
        self.model_train_timesteps = 0

    @torch.no_grad()
    def eval_data(self, data, update_elite_models=False):
        obs_list, action_list, next_obs_list, reward_list = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data)
        obs_list = torch.Tensor(obs_list)
        action_list = torch.Tensor(action_list)
        next_obs_list = torch.Tensor(next_obs_list)
        reward_list = torch.Tensor(reward_list)
        delta_obs_list = next_obs_list - obs_list
        obs_list, action_list = self.transform_obs_action(obs_list, action_list)
        model_input = torch.cat([obs_list, action_list], dim=-1).to(util.device)
        predictions = functional.minibatch_inference(args=[model_input], rollout_fn=self.model.predict,
                                                     batch_size=10000,
                                                     cat_dim=1)  # the inference size grows as model buffer increases
        groundtruths = torch.cat((delta_obs_list, reward_list), dim=1).to(util.device)
        eval_mse_losses, _ = self.model_loss(predictions, groundtruths, mse_only=True)
        if update_elite_models:
            elite_idx = np.argsort(eval_mse_losses.cpu().numpy())
            self.model.elite_model_idxes = elite_idx[:self.model.num_elite]
        return eval_mse_losses.detach().cpu().numpy(), None

    def reset_normalizers(self):
        self.obs_normalizer.reset()
        self.act_normalizer.reset()

    def update_normalizer(self, obs, action):
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(action)

    def transform_obs_action(self, obs, action, next_obs=None):
        obs = self.obs_normalizer.transform(obs)
        action = self.act_normalizer.transform(action)
        if next_obs == None:
            return obs, action
        elif next_obs != None:
            next_obs = self.obs_normalizer.transform(next_obs)
            return obs, action, next_obs
        
    def _sync_weight(self):
        for o, n in zip(self.true_valnet_old.parameters(), self.true_valnet.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        
        for i in range(self.model.ensemble_num):
            model_valnet_old = self.model_valnet_olds[i]
            model_valnet = self.model_valnets[i]
            for o, n in zip(model_valnet_old.parameters(), model_valnet.parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def update(self, data_batch, update_valnet=True):
        obs_batch, action_batch, next_obs_batch, reward_batch = \
            itemgetter("observations", 'actions', 'next_observations', 'rewards')(data_batch)
        obs_batch = torch.Tensor(obs_batch).to(util.device)
        action_batch = torch.Tensor(action_batch).to(util.device)
        next_obs_batch = torch.Tensor(next_obs_batch).to(util.device)
        reward_batch = torch.Tensor(reward_batch).to(util.device)

        delta_obs_batch = next_obs_batch - obs_batch
        obs_batch, action_batch, next_obs_batch = self.transform_obs_action(obs_batch, action_batch, next_obs_batch)

        # predict with model
        model_input = torch.cat([obs_batch, action_batch], dim=-1).to(util.device)
        predictions = self.model.predict(model_input)
        
        # compute the true value
        true_value = self.true_valnet(obs_batch)
        true_next_value = self.true_valnet_old(next_obs_batch)
        true_value_target = reward_batch + self.gamma * true_next_value 
        
        # compute the value learned from the model
        pred_means, _ = predictions
        obs_batch_tile = obs_batch.tile((self.model.ensemble_num, 1, 1))
        pred_next_obs = obs_batch_tile + pred_means[..., :-1]
        pred_next_values = [net(ip) for ip, net in zip(torch.unbind(pred_next_obs), self.model_valnet_olds)]
        pred_next_value = torch.stack(pred_next_values)
        pred_value_target = pred_means[..., -1:] + self.gamma * pred_next_value
        pred_values = [net(ip) for ip, net in zip(torch.unbind(obs_batch_tile), self.model_valnets)]
        pred_value = torch.stack(pred_values)
        
        if update_valnet:
            true_value_loss, model_value_loss = self.valnet_loss(
                true_value, true_value_target, pred_value, pred_value_target)

            # Compute true value networks' loss
            true_valnet_loss = true_value_loss
            # Update parameters
            self.true_val_optimizer.zero_grad()
            true_valnet_loss.backward()
            self.true_val_optimizer.step()
        
            model_valnet_loss = torch.sum(model_value_loss)
            # Update parameters
            self.model_val_optimzer.zero_grad()
            model_valnet_loss.backward(retain_graph=True)
            self.model_val_optimzer.step()

        # compute training loss
        groundtruths = torch.cat((delta_obs_batch, reward_batch), dim=-1).to(util.device)
        (
            train_mse_losses, 
            train_var_losses, 
            V_loss
        ) = self.model_loss(
            predictions, groundtruths, true_value_target, pred_value_target)
        
        train_mse_loss = torch.sum(train_mse_losses)
        train_var_loss = torch.sum(train_var_losses)
        train_val_loss = torch.sum(V_loss)
        train_transition_loss = train_mse_loss + train_var_loss + 0.35 * train_val_loss
        train_transition_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(self.model.min_logvar)  # why
        if self.use_weight_decay:
            decay_loss = self.model.get_decay_loss()
            train_transition_loss += decay_loss
        else:
            decay_loss = None
            
        # update transition model
        self.model_optimizer.zero_grad()
        train_transition_loss.backward()
        self.model_optimizer.step()
        
        self._sync_weight()
        
        # compute test loss for elite model
        return {
            "loss/train_model_loss_mse": train_mse_loss.item(),
            "loss/train_model_loss_var": train_var_loss.item(),
            "loss/train_model_loss": train_var_loss.item(),
            "loss/decay_loss": decay_loss.item() if decay_loss is not None else 0,
            "misc/max_std": self.model.max_logvar.mean().item(),
            "misc/min_std": self.model.min_logvar.mean().item()
        }

    def model_loss(
        self, predictions, groundtruths, true_value_target=None, 
        pred_value_target=None, mse_only=False
        ):
        pred_means, pred_logvars = predictions
        if self.inc_var_loss and not mse_only:
            # Average over batch and dim, sum over ensembles.
            inv_var = torch.exp(-pred_logvars)
            mse_losses = torch.mean(torch.mean(torch.pow(pred_means - groundtruths, 2) * inv_var, dim=-1), dim=-1)
            var_losses = torch.mean(torch.mean(pred_logvars, dim=-1), dim=-1)
        elif mse_only:
            mse_losses = torch.mean(torch.pow(pred_means - groundtruths, 2), dim=(1, 2))
            var_losses = None
        else:
            assert 0
        
        if true_value_target != None:
            true_value_tile = true_value_target.view(1, -1, 1).tile((self.model.ensemble_num, 1, 1))
            V_loss = torch.mean(torch.mean((true_value_tile.detach() - pred_value_target)**2, dim=-1), dim=-1)
            
            true_value = torch.mean(torch.mean(torch.abs(true_value_tile.detach()), dim=-1), dim=-1).sum()
            pred_value = torch.mean(torch.mean(torch.abs(pred_value_target.detach()), dim=-1), dim=-1).sum()
            # print("true value: ", true_value.item())
            # print("pred_value: ", pred_value.item())
            
            return mse_losses, var_losses, V_loss
        
        elif true_value_target == None:
            return mse_losses, var_losses

    def valnet_loss(self, true_value, true_value_target, pred_value, pred_value_target):
        # Compute the loss of true value networks
        # true_value_loss = torch.mean(F.relu(true_value_target.detach() - true_value, inplace=False)**2)
        true_value_loss = torch.mean((true_value_target.detach() - true_value)**2)
        # Average over batch and dim, sum over ensembles.
        model_value_loss = torch.mean(torch.mean((pred_value - pred_value_target.detach())**2, dim=-1), dim=-1)
            
        return true_value_loss, model_value_loss

    @torch.no_grad()
    def predict(self, obs, act, deterministic=False):
        """
        predict next_obs and rew
        """
        if len(obs.shape) == 1:
            obs = obs[None, ]
            act = act[None, ]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(util.device)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(util.device)

        scaled_obs, scaled_act = self.transform_obs_action(obs, act)

        model_input = torch.cat([scaled_obs, scaled_act], dim=-1).to(util.device)
        pred_diff_means, pred_diff_logvars = self.model.predict(model_input)
        pred_diff_means = pred_diff_means.detach().cpu().numpy()
        # add curr obs for next obs
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        ensemble_model_stds = pred_diff_logvars.exp().sqrt().detach().cpu().numpy()

        if deterministic:
            pred_diff_means = pred_diff_means
        else:
            pred_diff_means = pred_diff_means + np.random.normal(size=pred_diff_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = pred_diff_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        pred_diff_samples = pred_diff_means[model_idxes, batch_idxes]

        next_obs, rewards = pred_diff_samples[:, :-1] + obs, pred_diff_samples[:, -1]
        terminals = self.static_fns.termination_fn(obs, act, next_obs)

        # penalty rewards
        penalty_coeff = 1
        penalty_learned_var = True
        if penalty_coeff != 0:
            if not penalty_learned_var:
                ensemble_means_obs = pred_diff_means[:, :, :-1]
                mean_obs_means = np.mean(ensemble_means_obs, axis=0)  # average predictions over models
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0, :obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
                penalty = np.max(dists, axis=0)  # max distances over models
            else:
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
            penalized_rewards = rewards - penalty_coeff * penalty
        else:
            penalized_rewards = rewards

        assert (type(next_obs) == np.ndarray)
        info = {'penalty': penalty, 'penalized_rewards': penalized_rewards}
        penalized_rewards = penalized_rewards[:, None]
        terminals = terminals[:, None]
        return next_obs, penalized_rewards, terminals, info

    def update_best_snapshots(self, val_losses):
        updated = False
        for i in range(len(val_losses)):
            current_loss = val_losses[i]
            best_loss = self.best_snapshot_losses[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > 0.01:
                self.best_snapshot_losses[i] = current_loss
                self.save_model_snapshot(i)
                updated = True
                improvement = (best_loss - current_loss) / best_loss
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))
        return updated

    def reset_best_snapshots(self):
        self.model_best_snapshots = [deepcopy(self.model.ensemble_models[idx].state_dict()) for idx in
                                     range(self.model.ensemble_size)]
        self.best_snapshot_losses = [1e10 for _ in range(self.model.ensemble_size)]

    def save_model_snapshot(self, idx):
        self.model_best_snapshots[idx] = deepcopy(self.model.ensemble_models[idx].state_dict())

    def load_best_snapshots(self):
        self.model.load_state_dicts(self.model_best_snapshots)

    def save_model(self, info):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(info))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)

    def load_model(self, info):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_dir = os.path.join(save_dir, "ite_{}".format(info))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(model_save_dir, network_name + ".pt")
            torch.save(network, save_path)
