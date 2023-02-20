import gym
import numpy as np
import torch
#import wandb
import logging
import os
import time


import argparse
import pickle
import random
import sys

from CTDT.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_rtg_ct
from CTDT.models.decision_transformer import DecisionTransformer
from CTDT.models.ctdt import CTDT
from CTDT.models.mlp_bc import MLPBCModel
from CTDT.training.act_trainer import ActTrainer
from CTDT.training.seq_trainer import SequenceTrainer,SequenceTrainer_CT
from env.sim_env import newEnv

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    #log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    
    max_ep_len = variant['timeout']
    env = newEnv(timeout=max_ep_len)
    max_time = 25
    env_targets = [max_time*1,max_time*2,max_time*4,max_time*8,max_time*12,max_time*16,max_time*18, max_time*max_ep_len]  # evaluation conditioning targets
    scale = 100.  # normalization for rewards/returns

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = 1
    act_dim = 2
    act_noT_dim = act_dim - 1

    # make logger here
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    work_dir = os.path.join('./work_dir',
                                time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    time_prefix = time.strftime("%H:%M:%S", time.localtime())
    fHandler = logging.FileHandler(work_dir + '/'+time_prefix+'_'+env_name+'_'+dataset+ \
                                        '_'+variant['suffix']+'_'+variant['model_type']+'-log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    # log meta-data
    logger.info(variant)




    # load dataset
    dataset_path = f'data/{env_name}-{dataset}.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0).reshape(state_dim), np.std(states, axis=0).reshape(state_dim) + 1e-6

    num_timesteps = sum(traj_lens)

    logger.info('=' * 50)
    logger.info(f'Starting new experiment: {env_name} {dataset}')
    logger.info(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logger.info(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logger.info(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logger.info('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K,discrete_time=True):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        s, a, r, d, rtg, timesteps, mask, dt, turns = [], [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            #si = random.randint(0, traj['rewards'].shape[0] - 1)
            # for sim exp, we hard code the si to be 0
            si = 0
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_noT_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            dt.append(traj['dts'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))

            turns.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            turns[-1][turns[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

            timesteps.append(traj['times'][si:si + max_len].reshape(1, -1))
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_noT_dim)) * -10., a[-1]], axis=1)
            dt[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)) * -10., dt[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            turns[-1] = np.concatenate([np.zeros((1, max_len - tlen)), turns[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        dt = torch.from_numpy(np.concatenate(dt, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        # applicable for vanilla DT and BC
        if discrete_time:
            # round then convert to long
            timesteps = torch.from_numpy(np.round(np.concatenate(timesteps, axis=0))).to(dtype=torch.long, device=device)
            a = torch.cat([a,dt],dim=-1)
            return s, a, r, d, rtg, timesteps, mask
        else:
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.float32, device=device)
            return s, a,dt, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    elif model_type == 'ctdt':
                        ret, length = evaluate_episode_rtg_ct(
                            env,
                            state_dim,
                            act_noT_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                'target_rew': target_rew,
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len*max_time+1,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            action_tanh=False,
        )
        model_save_path = work_dir + '/'+env_name+'_'+dataset+'_'+variant['suffix']+'_DT.pt'
    elif model_type == 'ctdt':
        model = CTDT(
            state_dim=state_dim,
            act_dim=act_noT_dim,
            max_length=K,
            num_types = 4,
            max_ep_len=max_ep_len*max_time+1,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            action_tanh=False,
        )
        model_save_path = work_dir + '/'+env_name+'_'+dataset+'_'+variant['suffix']+'_CTDT.pt'
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
        model_save_path = work_dir + '/'+env_name+'_'+dataset+'_'+variant['suffix']+'_BC.pt'
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            save_path = model_save_path,
        )
    elif model_type == 'ctdt':
        trainer = SequenceTrainer_CT(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=None,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            save_path = model_save_path,
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            save_path = model_save_path,
        )

    

    if variant['train_model']:
        logger.info("begin model training")
        if variant['seed'] is not None:
            seednum = variant['seed']
            random.seed(seednum)
            np.random.seed(seednum)
            torch.manual_seed(seednum)
        # main training iteration here
        for iter in range(variant['max_iters']):
            outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=False)
            logger.info('=' * 80)
            logger.info(f'Iteration {iter+1}')
            for k, v in outputs.items():
                logger.info(f'{k}: {v}')
    
    else:
        if variant['model_path']:
            model_save_path = variant['model_path']
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            logger.info("no model loadded; using an untrained model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='unbiased')
    parser.add_argument('--dataset', type=str, default='default')  
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='ctdt')  # ctdt for continuous-time decision transformer, dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    #parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--train_model', action='store_true', help='train decision transformer')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--suffix', type=str, default='default')
    parser.add_argument('--timeout', type=int, default=20, 
            help='env timeout')
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
