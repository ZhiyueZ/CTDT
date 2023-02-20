
# synthetic data study logged data

import random
import numpy as np
from tqdm import tqdm
from env.sim_env import newEnv
import scipy.stats as stats
import pickle

import argparse
def readParser():
    parser = argparse.ArgumentParser(description='proof of concept experiment data generation')
    parser.add_argument('--seed', type=int, default=123, 
                        help='random seed (default: 123)')
    parser.add_argument('--num_traj_type1', type=int, default=5000, 
                        help='number of trajectories')
    parser.add_argument('--num_traj_type2', type=int, default=5000, 
                    help='number of trajectories')
    parser.add_argument('--timeout', type=int, default=20, 
                        help='environment timeout')
    parser.add_argument('--save', type=str, default='unbiased-default',
                    help='DT model name string')
    parser.add_argument('--type1_mean', type=float, default=0., 
                help='dosage mean')
    parser.add_argument('--type2_mean', type=float, default=2., 
                help='dosage mean')
    parser.add_argument('--biased', action='store_true', help='bias dt mean')     
    return parser.parse_args()


# dt model here
def h_boost(dosage,env):
    #below d_optim
    if dosage<=env.d_optim:
        slope = (env.h0-env.m)/env.d_optim
        h = env.m+slope*dosage
    # above d_optim
    else:
        slope = -(env.h0-env.m)/env.d_optim
        h = env.m+slope*(dosage-2*env.d_optim)
    return h
def rand_data(env,dosage_change,args):
        env.reset()
        total_reward = 0
        dosages = np.array([],dtype=np.float32)
        deltats = np.array([],dtype=np.float32)
        rewards = np.array([],dtype=np.float32)
        next_states = np.array([],dtype=np.float32)
        done = False
        dones = np.array([],dtype=bool)
        states = np.array([],dtype=np.float32)
        time_seq=np.array([],dtype=np.float32)

        while (not done):

            optimal_dosage = env.d_optim
            dosage_mean = optimal_dosage + dosage_change
            dosage = stats.norm.rvs(loc=dosage_mean, scale=1, size=1).item()
            h_exp = h_boost(dosage,env)
            optim_dt = h_exp/env.scale
            if args.biased:
                optim_dt = optim_dt - 3
            else:
                optim_dt = optim_dt 
            
            dt = stats.norm.rvs(loc=optim_dt, scale=5, size=1)
            #dosage = np.clip(dosage,0,None).item()
            dt = np.clip(dt,1e-6,None).item()
            action = np.array([dosage,dt])
            # record data pre step
            dosages = np.concatenate([dosages,[dosage]],axis=0)
            deltats = np.concatenate([deltats,[dt]],axis=0)
            states = np.concatenate([states,env.m.astype(np.float32)],axis=0)
            time_seq = np.concatenate([time_seq,[env.time]],axis=0)
            # execute the action
            next_state, reward, done, _= env.step(action)
            #record data post step
            rewards = np.concatenate([rewards,[reward]],axis=0)
            dones = np.concatenate([dones,[done]],axis=0)
            next_states = np.concatenate([next_states,next_state.astype(np.float32)],axis=0)
            total_reward += reward
        
        traj_data = {}
        traj_data['observations'] = states
        traj_data['next_observations'] = next_states
        traj_data['actions'] = dosages
        traj_data['dts'] = deltats
        traj_data['terminals'] = dones
        traj_data['rewards'] = rewards
        traj_data['times'] = time_seq
        return traj_data


def main(args=None):
    if args is None:
        args = readParser()
    seednum = args.seed
    random.seed(seednum)
    np.random.seed(seednum)
    save_path = 'data/'+args.save + '.pkl'
    env = newEnv(timeout=args.timeout)
    dataset=[]

    # populating dataset
    for i in tqdm(range(args.num_traj_type1)):
        traj = rand_data(env,args.type1_mean,args)
        dataset.append(traj)

    for i in tqdm(range(args.num_traj_type2)):    
        traj = rand_data(env,args.type2_mean,args)
        dataset.append(traj)


    with open(save_path, 'wb') as f:
        pickle.dump(dataset,f)
    






if __name__ == '__main__':
    main()