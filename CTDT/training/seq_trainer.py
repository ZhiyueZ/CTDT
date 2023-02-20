import numpy as np
import torch

from CTDT.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


class SequenceTrainer_CT(Trainer):

    def train_step(self):
        states, actions, dts, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size,discrete_time=False)
        action_target = torch.clone(actions)
        deltat_target = torch.clone(dts)


        state_preds, action_preds, deltat_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        deltat_preds = deltat_preds.reshape(-1)[attention_mask.reshape(-1) > 0]
        deltat_target = deltat_target.reshape(-1)[attention_mask.reshape(-1) > 0]
        
        d_loss = torch.mean((action_preds-action_target)**2)
        dt_loss = torch.mean((deltat_preds-deltat_target)**2)
        loss = d_loss + dt_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/deltaT_error'] = torch.mean((deltat_preds-deltat_target)**2).detach().cpu().item()
        return loss.detach().cpu().item()
