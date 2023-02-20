import numpy as np
import torch
import torch.nn as nn
import math
import transformers

from CTDT.models.model import TrajectoryModel
from CTDT.models.trajectory_gpt2 import GPT2Model


class CTDT(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            num_types = 4,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        # to account for temporal and type embeddings
        self.total_size = hidden_size*2
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.total_size, 
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)


        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_dt = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(self.total_size)
        self.d_time = round(hidden_size/2)
        self.embed_size = hidden_size - self.d_time
        self.div_term = torch.exp(torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)).reshape(1, 1, -1)
        self.embed_type = nn.Embedding(num_types, self.embed_size)
        
        
        self.predict_state = torch.nn.Linear(self.total_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.total_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_dt = torch.nn.Linear(self.total_size, 1)
    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        return pe

    # 0 for return 1 for state 2 for action 3 for dt
    def compute_type_embedding(self,vec_input,type_index):
        batch_size,seq_length = vec_input.shape[0],vec_input.shape[1]
        type_vec = torch.ones([batch_size,seq_length],device=vec_input.device)
        type_vec = (type_vec * type_index).to(dtype=torch.long)
        type_emb = self.embed_type(type_vec)
        return type_emb

    def forward(self, states, actions, dts, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        dt_embeddings = self.embed_dt(dts)
        time_embeddings = self.compute_temporal_embedding(timesteps)
        return_type_emb = self.compute_type_embedding(returns_to_go,0)
        state_type_emb = self.compute_type_embedding(states,1)
        action_type_emb= self.compute_type_embedding(actions,2)
        dt_type_emb = self.compute_type_embedding(dts,3)

        # time embeddings are concatenated, rather than added, to the values
        # together with the type embeddings
        state_embeddings = torch.cat([state_embeddings,time_embeddings,state_type_emb],-1)
        action_embeddings = torch.cat([action_embeddings,time_embeddings,action_type_emb],-1)
        returns_embeddings = torch.cat([returns_embeddings,time_embeddings,return_type_emb],-1)
        dt_embeddings  = torch.cat([dt_embeddings,time_embeddings,dt_type_emb],-1)
        # this makes the sequence look like (R_1, s_1, a_1,dt_1, R_2, s_2, a_2,dt_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings, dt_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.total_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 4*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), actions (2), or dts (3); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.total_size).permute(0, 2, 1, 3)

        # get predictions
        dt_preds = self.predict_dt(x[:,2])  # predict dt given state and treatment
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, dt_preds

    def get_action(self, states, actions, dts, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        dts = dts.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            dts = dts[:,-self.max_length:]
            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.float32)
            dts = torch.cat(
                [torch.zeros((dts.shape[0], self.max_length-dts.shape[1], 1), device=dts.device), dts],
                dim=1).to(dtype=torch.float32)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, dts, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        action_pred = action_preds[0,-1]
        # replace the last action
        actions[0,-1]=action_pred
        _, _, dt_preds = self.forward(
            states, actions, dts, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        dt_pred = dt_preds[0,-1]
        output_action = torch.cat([action_pred,dt_pred],dim=-1) 
        return output_action
